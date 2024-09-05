from datetime import timedelta
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel
from shapely.geometry import LineString, Point
from uszipcode.search import SearchEngine

ZipCode = str
Locatable = Union[Point, tuple[float, float], str, ZipCode]

search = SearchEngine()


class RoadClass(Enum):
    # Freeway (Multi-lane, controlled access)
    FREEWAY = 1
    # Primary Provincial/Territorial highway
    PP_TH = 2
    # Secondary Provincial/territorial highway/ municipal arterial
    SP_TH_MA = 3
    # Municipal collector/Unpaved secondary provincial/territorial highway
    MC_USP_TH = 4
    # Local street/ winter road
    LS_WR = 5
    # NOTE: Ferry not used for routing
    # FERRY = 6


class Road(BaseModel):
    start: Point
    end: Point
    juris_name: Optional[str]
    road_class: Optional[RoadClass]
    surface: Optional[Literal["PAVED", "UNPAVED", ""]]
    lanes: Optional[int]
    speed_limit: Optional[int] # NOTE: km/h
    distance: Optional[float] # NOTE: km
    admin: Optional[str] # NOTE: Administrative region, Province/Territory
    road_name: Optional[str]
    coords: Optional[LineString]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_shp_edge_attr(
        cls,
        start: Point,
        end: Point,
        shp: dict[str, Any],
        coords: Optional[LineString]=None
    ) -> 'Road':

        return cls(
            start=start,
            end=end,
            juris_name=shp.get("JURISNAME"),
            coords=coords,
            road_name=shp.get("ROADNAME"),
            road_class=RoadClass(shp.get("CLASS")),
            surface=shp.get("SURFACE", "").upper(),
            lanes=shp.get("LANES"),
            speed_limit=shp.get("SPEEDLIM"),
            distance=shp.get("LENGTH"),
            admin=shp.get("ADMIN")
        )

    def __str__(self) -> str:
        _str = f"{self.start} -> {self.end} ({self.distance} km)"
        if self.road_name is not None: _str += f" {self.road_name},"
        if self.road_class is not None: _str += f" {self.road_class.name},"
        if self.speed_limit is not None: _str += f" {self.speed_limit} km/h"
        return _str


class Location(BaseModel):
    point: Point
    zip_code: Optional[ZipCode] = None
    admin: Optional[str] = None
    name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_city(cls, city: str, state: Optional[str]=None) -> 'Location':
        matched_zc = search.by_city(city) if state is None else search.by_city_and_state(city, state)
        if len(matched_zc) == 0: raise ValueError(f"No zipcode found for {city}, {state}")
        zipcode = matched_zc[0]
        return cls._from_zc_obj(zipcode)

    @classmethod
    def from_zipcode(cls, zipcode: ZipCode) -> 'Location':
        zipcode = search.by_zipcode(zipcode)
        return cls._from_zc_obj(zipcode)

    @classmethod
    def from_locatable(cls, locatable: Any) -> 'Location':
        if isinstance(locatable, Point):
            return cls(point=locatable)
        elif isinstance(locatable, tuple):
            if len(locatable) != 2: raise ValueError("Invalid locatable tuple")
            return cls(point=Point(*locatable))
        elif isinstance(locatable, str):
            # check if locatable is a zipcode
            if locatable.isdigit() and len(locatable) == 5:
                return cls.from_zipcode(locatable)
            return cls.from_city(locatable)
        else:
            raise ValueError("Invalid locatable")

    @classmethod
    def _from_zc_obj(cls, zipcode: dict[str, Any]) -> 'Location':
        return cls(
            point=Point(zipcode.lng, zipcode.lat),
            zip_code=zipcode.zipcode,
            admin=zipcode.state,
            name=zipcode.city
        )


class Route(BaseModel):
    roads: list[Road]

    @property
    def _approx_coords(self) -> LineString:
        points = [self.roads[0].start]
        for road in self.roads[1:]:
            points.append(road.end)
        return LineString(points)

    @property
    def duration(self) -> timedelta:
        duration = timedelta(0)
        for road in self.roads:
            distance = road.distance or road.start.distance(road.end)
            speed = road.speed_limit or 100
            duration += timedelta(hours=distance / speed)
        return duration

    @property
    def coords(self) -> LineString:
        if not all(road.coords is not None for road in self.roads):
            return self._approx_coords

        coords = []
        for road in self.roads:
            if road.coords is None: continue
            coords.extend(road.coords)
        return coords

    @property
    def total_distance(self) -> float:
        return sum(road.distance for road in self.roads)

    def __str__(self) -> str:
        return "\n".join([str(road) for road in self.roads])
