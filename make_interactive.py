from bokeh.plotting import figure, curdoc
from bokeh.layouts import column
from bokeh.models import CheckboxGroup, ColumnDataSource, Slider
import numpy as np
from utils import * 


# Generate example data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x)
data = {'x': x, 'y': y}


# Load up data
my_file_name = './products/run0.npz'
data = np.load(my_file_name, allow_pickle=True)
redshifts = data['redshifts']
metric_tracker = data['metric_tracker']
metric_list = data['metric_list']
flags = np.asarray(list(metric_list.item().keys()), dtype=str)
efficiencies = calc_efficiences(metric_tracker)


bokeh_data = {'x': redshifts, 
              'y': efficiencies}

# Create ColumnDataSource
source = ColumnDataSource(data=bokeh_data)

# Create initial plot
plot = figure(title="Fraction of Positive Computations",
              plot_height=400, plot_width=800)

# Plot the data
plot.line('x', 'y', source=source, line_width=2)

# Create CheckboxGroup for selecting flags
checkbox_group = CheckboxGroup(labels=list(flags), active=[], width=100)

# Function to update the plot based on checkbox selections
def update(attr, old, new):
    selected_flags = [flags[i] for i in checkbox_group.active]
    gind = np.where(selected_flags == flags)
    print('hiiiii')
    positive_fraction = compute_fraction(selected_flags)
    plot.title.text = f"Fraction of Positive Computations: {positive_fraction:.2f}"

# Compute fraction of positive computations based on selected flags
def compute_fraction(selected_flags):
    gind = np.where(selected_flags == flags)
    print(gind)

    positive_fraction = np.zeros(len(efficiencies))
    for z_ind in np.arange(len(efficiencies)):
        still_positive = True
        for j, selected_flag in enumerate(selected_flags):
            print('hello',metric_tracker[z_ind, j])
            if selected_flag & positive_fraction:
                still_positive = True
            else:
                still_positive = False
            positive_fraction
    return still_positive
    #if 'Flag 1' in selected_flags and 'Flag 2' in selected_flags:
        #positive_fraction = np.sum((metric_tracker[])


# Add callback to checkbox group
checkbox_group.on_change('active', update)

# Add slider for another variable
slider = Slider(start=0, end=10, value=5, step=0.1, title="Another Variable")

# Layout setup
layout = column(checkbox_group, plot, slider)

# Add layout to current document
curdoc().add_root(layout)
