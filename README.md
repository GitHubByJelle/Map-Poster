# Creating a Map Poster
This project is a personal experiment inspired by the poster-style map artworks often advertised on social media. After seeing these ads repeatedly, I set myself the challenge to create a similar system from scratch, not just replicating the idea, but also making it configurable and extensible.

The project leverages OpenStreetMap (OSM) data to extract information about a predefined region of interest. Using this data, I implemented a pipeline that visualizes the map with customizable features such as color schemes, plotted elements, and layer inclusion/exclusion. The maps are rendered using Matplotlib, with a focus on achieving high enough resolution and quality for use in poster-sized prints.

This project was developed in a single afternoon as a way to create something creative with my programming skills. Even though it is “just” an afternoon project, my goal is to make it possible for anyone to easily create their own poster as well, without cost or complexity. Future improvements could focus on better formatted code and adding interactive previews through a frontend interface.

<p align="center" width="100%">
    <img src="src/images/poster.png" alt="Example Map Poster" width="70%">
</p>

# Implementation Details
The code is written in Python and relies on the packages described in the `pyproject.toml`. The most important packages used are:
* osmnx
* matplotlib

# How to use
This project uses [uv](https://astral.sh/) as the package manager. Begin by installing the required dependencies:
```bash
uv sync
```

Next, configure the settings file according to your needs. You can configure what area of the map to render (by place name, center + radius, or bounding box), which layers to draw, and how the map looks through colors, widths, and figure options. Output settings let you control the file format, title, coordinates, and styling details. For full documentation of all available fields and defaults, see `src/models.py`.

Once configured, generate a map with:
```bash
uv run src/make_map_poster.py ./src/config/config.example.yml
```