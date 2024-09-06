# US Routing

US Routing is a Python library for fast local routing in the United States. It's useful when approximations are acceptable. It bootstraps from the [North American Roads dataset](https://geodata.bts.gov/datasets/usdot::north-american-roads).

## Installation

You can install US Routing using pip:

```sh
pip install us-routing
```

or using poetry:

```sh
git clone https://github.com/ivanbelenky/us-routing.git
cd us-routing
poetry install
```

## Usage

Here's a quick example of how to use US Routing:

```python
from us_routing import get_route

# Route between two cities
r = get_route('New York', 'Los Angeles', edge_distance="DURATION")
print(r.total_distance, r.duration)
# Output in km of course:
# 4434.759999999997 1 day, 20:46:24.499959

# Print route steps
print(r)
# Output: [
#POINT (-73.99811899964011 40.7508730002449) -> POINT (-74.0013209995546 40.74648499998924) (0.5700000000000001 km) 9TH AV, SP_TH_MA, 72 km/h
#POINT (-74.0013209995546 40.74648499998924) -> POINT (-74.0054249996425 40.74097199980971) (0.6799999999999999 km) 9TH AV, SP_TH_MA, 72 km/h
#POINT (-74.0054249996425 40.74097199980971) -> POINT (-74.00819599956175 40.74211600011984) (0.27 km) W 14TH ST, SP_TH_MA, 72 km/h
#POINT (-74.00819599956175 40.74211600011984) -> POINT (-74.0090509998795 40.74090099973697) (0.16 km) 10TH AV, SP_TH_MA, 72 km/h
# ...]

# Route between zip codes
r = get_route('10001', '60007')
print(r.total_distance, r.duration)
# Output:
# 1315.910000000001 13:20:26.827392


# Route between coordinates, will raise ValueError if closest node in the graph is too far from the location
try:
    r = get_route((40.7128, -74.0060), (34.0522, -118.2437), d_threshold=0.00001)
    print(r.total_distance, r.duration)
except ValueError as e:
    print(e)
#  Node 2bd87209-d2fe-4f41-a89f-29104aeb5cf9 is too far from location point=<POINT (40.713 -74.006)> zip_code=None admin=None name=None

r = get_route((-74.0060, 40.7128), (-118.2437, 34.0522), d_threshold=10)


```

## Features

- Fast routing between US locations (cities, zip codes, or coordinates)
- Multiple routing options by default (shortest distance, fastest time)
- Detailed route information (distance, duration, states traversed)

## Data Source

The routing data is based on the North American Roads dataset. The library includes functionality to download and process this data:


```python
from us_routing import download_north_american_roads

download_north_american_roads()
```

## Development

To set up the development environment:

1. Clone the repository
2. Install Poetry: `pip install poetry`
3. Install dependencies: `poetry install`

### Custom Routers

This package provides a `BaseRouter` class that you can use to build your own custom routing graphs. It exposes a very simple API to create routers from shapefiles containing multiple geometries and optional attributes for those geometries. It looks something like this:

```python
geometries: Sequence[LineString | MultiLineString]
geometries_data: Optional[Sequence[Dict[str, Any]]] = None # should be serializable

# Optional arguments to define how attributes are compared and set to the edges of the graph
edge_attr_equal: Optional[Callable[[Any, Any], bool]] = ...
edge_attr_setter: Optional[Callable[[Any, Any], dict[str, Any]]] = ...

router = BaseRouter.from_geometries(geometries, geometries_data, edge_attr_equal, edge_attr_setter)
router.serialize("path_to_save")
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
