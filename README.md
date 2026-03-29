# PlotDeck

- Import a .csv file with rows of time-series data. The first row must contain variable names for each row. Separate/group variavles with '.' and this tool will automatically group items for easier navigation through variables.

- Select an x-axis variable for all plots. Then select y-axis variables to throw up on each sub plot.

- Save plot sets and load them again later.

- WASD navigation for pan/zoom of all plots. X-axis across all subplots remains synced.

- Export current plot view as .png image.


**To run:** `python PlotDeck.py`

## Requirements:
- pandas
- numpy
- PyQt5
- pyqtgraph