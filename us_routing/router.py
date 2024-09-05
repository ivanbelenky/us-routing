import copy
import pickle
import uuid
import warnings
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import networkx as nx
import numpy as np
from rtree import index
from shapely.geometry import LineString, MultiLineString, Point
from tqdm import tqdm

from .domain import Locatable, Location, Road, Route

ZipCode = str
EdgeWeightFunction = Callable[[Any, Any, dict[str, Any]], float]


class BaseRouter:
    def __init__(
        self,
        graph: Optional[nx.Graph]=None,
        **kwargs,
    ) -> None:
        '''Base Router class acting as proxy of the graph object'''

        self.graph = graph
        tree = kwargs.get('tree', None)
        self.tree = tree if tree is not None else self._create_node_tree() if graph is not None else None

    def _create_node_tree(self) -> index.Index:
        nodes_points = {nid: self.graph.nodes[nid]['point'] for nid in self.graph.nodes}
        tree = index.Index()
        for nid, node in nodes_points.items():
            tree.insert(int(nid), node.coords[0], nid)
        return tree

    def get_closest_node_id(self, point: Point) -> uuid.UUID:
        return self.tree.nearest(point.coords[0], 1, objects=True).__next__().object

    def _validate_distances(self, nodes: list[tuple[uuid.UUID, Location]], threshold: float) -> None:
        for node, loc in nodes:
            if np.linalg.norm(np.array(loc.point.coords[0]) - np.array(self.graph.nodes[node]['point'].coords[0])) > threshold:
                raise ValueError(f"Node {node} is too far from location {loc}")

    def route(
        self,
        start: Locatable,
        end: Locatable,
        edge_distance: Optional[Callable[[Any], float]]=None,
        threshold: Optional[float]=10.,
    ) -> Route:

        start, end = Location.from_locatable(start), Location.from_locatable(end)
        start_node = self.get_closest_node_id(start.point)
        end_node = self.get_closest_node_id(end.point)
        self._validate_distances([(start_node, start), (end_node, end)], threshold)
        path = self._route(start_node, end_node, edge_distance)
        roads = []
        for i in range(0, len(path)-1):
            roads.append(
                Road.from_shp_edge_attr(
                    start=self.graph.nodes[path[i]]['point'],
                    end=self.graph.nodes[path[i+1]]['point'],
                    shp=self.graph[path[i]][path[i+1]]
                )
            )

        return Route(roads=roads)

    def _route(
        self,
        start_node: uuid.UUID,
        end_node: uuid.UUID,
        edge_distance: Optional[Callable[[Any, Any, dict[str, Any]], float] | str]=None
    ) -> list[uuid.UUID]:
        return nx.shortest_path(self.graph, source=start_node, target=end_node, weight=edge_distance)

    def iter_routes(
        self,
        start: Locatable,
        end: Locatable,
        weight: Optional[EdgeWeightFunction]=None,
        max_routes: Optional[int]=None,
        max_route_distance: Optional[float]=None,
        max_route_duration: Optional[timedelta]=None,
    ) -> Iterator[Route]:

        start_node = self.get_closest_node_id(start)
        end_node = self.get_closest_node_id(end)
        paths = nx.shortest_simple_paths(self.graph, source=start_node, target=end_node, weight=weight)
        route_count = 0
        for path in paths:
            route = Route(roads=[
                Road.from_shp_edge_attr(
                    start=self.graph.nodes[path[i]]['point'],
                    end=self.graph.nodes[path[i+1]]['point'],
                    shp=self.graph[path[i]][path[i+1]]
                ) for i in range(0, len(path)-1)
            ])

            if max_route_distance and route.total_distance > max_route_distance: break
            if max_route_duration and route.duration > max_route_duration: break
            if max_routes and route_count >= max_routes: break
            yield route

            route_count += 1

    @classmethod
    def from_geometries(
        cls,
        geometries: list[LineString | MultiLineString],
        geometries_data: Optional[list[dict[str, Any]]],
        edge_attr_equal: Optional[Callable[[Any, Any], bool]]=lambda x, y: x==y,
        edge_attr_setter: Optional[Callable[[Any, Any], dict[str, Any]]] = lambda x, _: x,
    ) -> 'BaseRouter':
        if geometries_data and len(geometries) != len(geometries_data):
            raise ValueError("Geometries and geometries_data must have the same length")

        for geometry, data in zip(geometries, geometries_data):
            if isinstance(geometry, MultiLineString):
                for line in geometry.geoms:
                    geometries.append(line)
                    geometries_data.append(data)
        geometries = {i: geometry for i, geometry in enumerate(geometries)
                      if not isinstance(geometry, MultiLineString)}
        geometries_data = {i: data for i, data in enumerate(geometries_data)
                           if not isinstance(data, MultiLineString)}

        G = cls.create_graph_from_geometries(geometries, geometries_data, edge_attr_equal, edge_attr_setter)
        return cls(graph=G)

    @classmethod
    def create_graph_from_geometries(
        cls,
        geometries: dict[int, LineString],
        geometries_data: dict[int, dict[str, Any]],
        edge_attr_equal: Optional[Callable[[Any, Any], bool]]=lambda x, y: x==y,
        edge_attr_setter: Optional[Callable[[Any, Any], dict[str, Any]]] = lambda x, _: x,
    ) -> nx.Graph:

        tree = index.Index()
        for geom_id, geometry in tqdm(geometries.items(), desc="Creating tree"):
            tree.insert(geom_id, geometry.bounds)

        other_intersections = {}
        road_to_node_at_index = defaultdict(list)
        nodes_to_id = {}

        # Calculates intersections between geometries and creates nodes for start
        # and end points of each road as well as intersection points.
        for i, geometry in tqdm(geometries.items(), desc="Calculating intersections"):
            intersections = [ix for ix in list(tree.intersection(geometry.bounds)) if ix != i]

            start_point, end_point = Point(geometry.coords[0]), Point(geometry.coords[-1])
            for point, idx in zip((start_point, end_point), (0, len(geometry.coords)-1)):
                if point not in nodes_to_id: nodes_to_id[point] = uuid.uuid4()
                road_to_node_at_index[i].append((nodes_to_id[point], point, idx))

            for j in intersections:
                if i==j or not geometries[i].intersects(geometries[j]): continue
                intersection_point = geometry.intersection(geometries[j])

                if not isinstance(intersection_point, Point):
                    other_intersections[intersection_point.geom_type] = (frozenset((i, j)))
                    continue

                distances_i = np.linalg.norm(np.array(geometry.coords) - np.array(intersection_point.coords), axis=-1)
                distances_j = np.linalg.norm(np.array(geometries[j].coords) - np.array(intersection_point.coords), axis=-1)
                index_of_closest_i, index_of_closest_j = np.argmin(distances_i), np.argmin(distances_j)

                if intersection_point not in nodes_to_id: nodes_to_id[intersection_point] = uuid.uuid4()

                road_to_node_at_index[i].append((nodes_to_id[intersection_point], intersection_point, index_of_closest_i))
                road_to_node_at_index[j].append((nodes_to_id[intersection_point], intersection_point, index_of_closest_j))

        # Creates graph from nodes and edges.
        # "reversal" roads are nodes, and intersections are edges. Down below
        # that gets inverted
        graph = nx.Graph()
        for point, nid in tqdm(nodes_to_id.items(), desc="Adding nodes"):
            graph.add_node(nid, point=point)

        for road, nodes_on_road in tqdm(road_to_node_at_index.items(), desc="Adding edges"):
            nodes_on_road = list(set(nodes_on_road))
            nodes_on_road.sort(key=lambda node: node[2])
            if len(nodes_on_road) < 2: warnings.warn("less than two nodes per road segment, something is off")
            for i in range(0, len(nodes_on_road) - 1):
                graph.add_edge(nodes_on_road[i][0], nodes_on_road[i+1][0], **geometries_data[road])

        # Removes nodes with only two neighbors
        for i in range(0, 3):
            for node in tqdm(list(graph.nodes), desc=f"Reducing graph {i+1}-th round"):
                if len(list(graph.neighbors(node))) != 2: continue
                neighbor1, neighbor2 = list(graph.neighbors(node))
                e1, e2 = graph[neighbor1][node], graph[neighbor2][node]
                if not edge_attr_equal(e1, e2): continue
                edge_attr = edge_attr_setter(e1, e2)
                graph.add_edge(neighbor1, neighbor2, **edge_attr)
                graph.remove_edge(node, neighbor1)
                graph.remove_edge(node, neighbor2)
                graph.remove_node(node)

        # Removes isolated nodes
        for node in tqdm(list(graph.nodes), desc="Removing isolated nodes"):
            if len(list(graph.neighbors(node))) == 0: graph.remove_node(node)

        return graph

    def serialize(self, path: Path) -> tuple[list[Any], list[Any], index.Index]:
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)



class USRouter(BaseRouter):
    def __init__(self, graph: Optional[nx.Graph]=None, tree: Optional[index.Index]=None) -> None:
        super().__init__(graph=graph, tree=tree)

    @classmethod
    def from_cached(cls, class_lte: Optional[int]=1) -> 'USRouter':
        graph = nx.Graph()
        path = Path('./usrouter123.pkl') #TODO: change this, inspire from uszipcodes
        road_classes = "".join(range(1, class_lte+1))
        with open(path/f'usrouter{road_classes}.pkl', "rb") as f:
            nodes, edges = pickle.load(f)
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
        return cls(graph=graph)

    @staticmethod
    def _edge_attr_equal_deafult(e1: dict[str, Any], e2: dict[str, Any]) -> bool:
        for k in e1.keys():
            if k in ["ID", "LENGTH", "LINKID"]: continue
            if e1[k] != e2[k]: return False
        return True

    @staticmethod
    def _edge_attr_setter_default(e1: dict[str, Any], e2: dict[str, Any]) -> dict[str, Any]:
        edge_attrs = copy.deepcopy(e1)
        edge_attrs["ID"] = [e1["ID"], e2["ID"]]
        edge_attrs["LENGTH"] = e1["LENGTH"] + e2["LENGTH"]
        edge_attrs["LINKID"] = [e1["LINKID"], e2["LINKID"]]
        return edge_attrs


__us_router = USRouter(graph=None)#.from_cached()

def get_route(
        start: Location,
        end: Location,
        weight: Optional[EdgeWeightFunction]=None,
    ) -> Route:
    return __us_router.route(start, end, weight)


def all_routes(
        start: Location,
        end: Location,
        weight: Optional[EdgeWeightFunction]=None,
        max_routes: Optional[int]=None,
        max_route_distance: Optional[float]=None,
        max_route_duration: Optional[timedelta]=None,
    ) -> Iterator[Route]:
    return __us_router.iter_routes(start, end, weight, max_routes, max_route_distance, max_route_duration)
