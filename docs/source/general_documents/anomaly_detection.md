Anomaly Detection feedback
==========================

Anomaly detection feedback should have two functions:
1.	Labelling unlabeled anomalies
2.	Visualizing different categories of anomalies

However, the way to visualize both points can be done the same. In order to not reinvent the wheel, it is best to reuse the layout from the main Nereda screen.

For both, the left side of the screen contains a plot with the prediction, measurement and anomalies. Below that is a table where anomalies can be selected, which are then visualized in the earlier mentioned plot.

The right side of the screen should contain selectable options. You could keep the options for a date timeframe, location and goal variable, but add an option for the type of anomaly if you want to see specific categories of anomalies.

These are then shown in the table below where they can be selected. Clicking on a row selects them, and then on the right you should be able to select the type of anomaly and save that to a database. This also includes ‘overwriting’ previously labelled anomalies, since new categories might be added or changed. A new record is then stored and only the most recent label is viewable for clarity.

The following information should then be stored:
*	Trainingset
*	Pipeline version
*	Hyperparameters
*	Anomaly duration
*	Label
*	Start time anomaly
*	End time anomaly
*	Location
*	Goal variable
*	Time that the anomaly was stored

Remarks by Tim
--------------

A number of remarks:

*   What happens when something is not an anomaly anymore, because we use a new model?
*   Related: If the anomaly is linked to a pipeline version, do you have to label everything again upon deploying a new pipeline?
*   Will there be a discrete set of anomaly options, or will that be free text? If we later on want to use these labels for classification + prioritising, free text will not help?
*   Will we store the user that has labelled the anomaly? For traceability?
*   We store the changes of labelling, how can anyone see this history?

Technical:

*  How do we store that data, in MongoDB? Or in SQL? Or ...?
*   Is the training set stored with every anomaly? Why is that stored anyways?
*   What happens if a sensor/location is removed?

General:

*    Can we also label stuff that is an anomaly, but not detected by the model?
*    Can we also label significant changes that affect the data: e.g. new method of measuring?

We could also make some user stories for this to clarify what functionality should be build...

Remarks by Rutger
-----------------

I would separate the labeling (conceptually) completely from the used models. You should just be able to label data, from a start time to an end time.

In a UI you could use a model to pre-set interesting start/end times, and you could even save the fact that a model has detected an anomaly for a label, but I think these things should also be able to exists separately. It should be just another time series, but in this case labels (with possible metadata).
