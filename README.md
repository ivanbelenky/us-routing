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
from us_routing import route, all_routes

# Route between two cities
r = route('New York', 'Los Angeles')
print(r.distance, r.duration)
# Output: 2789.0 42.0

# Print route steps
print(r)
# Output: [
# Road('I-78 W',),
# Road('I-80 W',),
# ...]

# Route between zip codes
r = route('10001', '60007')
print(r.total_distance, r.duration)
# Output: 1160.0 17.0

# Route between coordinates
r = route((40.7128, -74.0060), (34.0522, -118.2437))
print(r.total_distance, r.duration)
# Output: 2789.0 42.0

# Get multiple routes

routes = all_routes('New York', 'Los Angeles', max_routes=3)
for i, r in enumerate(routes):
    print(r.total_distance, r.duration)

# Output:
# 2789.0 42.0
# 2853.0 43.0
# 2855.0 43.5
```

## Features

- Fast routing between US locations (cities, zip codes, or coordinates)
- Multiple routing options (shortest distance, fastest time)
- Detailed route information (distance, duration, states traversed)
- Support for finding alternative routes

## Data Source

The routing data is based on the North American Roads dataset. The library includes functionality to download and process this data:


```python

```

## Development

To set up the development environment:

1. Clone the repository
2. Install Poetry: `pip install poetry`
3. Install dependencies: `poetry install`


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
