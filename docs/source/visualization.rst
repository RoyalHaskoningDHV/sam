.. _visualization:

=============
Visualization
=============

This is the documentation for visualization functions.

Precision Recall curve plot
---------------------------
.. autofunction:: sam.visualization.plot_precision_recall_curve

.. image:: general_documents/images/precision_recall_curve.png

Autocorrelation plot
--------------------
.. autofunction:: sam.visualization.plot_lag_correlation

.. image:: general_documents/images/autocorrelation.png

Threshold curve plot
---------------------------
.. autofunction:: sam.visualization.plot_threshold_curve

.. image:: general_documents/images/threshold_curve.png

Incident heatmap plot
---------------------------
.. autofunction:: sam.visualization.plot_incident_heatmap

.. image:: general_documents/images/incident_heatmap.png

.. _flatline-removal-plot:

Flatline Removal plot
-----------------------
.. autofunction:: sam.visualization.diagnostic_flatline_removal

.. image:: general_documents/images/flatline_removal_example.png

.. _extreme-removal-plot:

Extreme value removal plot
---------------------------
.. autofunction:: sam.visualization.diagnostic_extreme_removal

.. image:: general_documents/images/extreme_values_example_testset.png

Quantile Regression plot
---------------------------
.. autofunction:: sam.visualization.sam_quantile_plot

.. image:: general_documents/images/quantile_plot.png

Feature importances plot
---------------------------
.. autofunction:: sam.visualization.plot_feature_importances

.. image:: general_documents/images/quantile_importances_barplot.png
.. image:: general_documents/images/quantile_importances_barplot_sum.png

Evaluate predict ahead
---------------------------
.. autofunction:: sam.visualization.performance_evaluation_fixed_predict_ahead
