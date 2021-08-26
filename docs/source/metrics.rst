.. _metrics:

=============
Metrics
=============

This is the documentation for Metrics functions.

Incident recall
---------------
A new metric as an extension of `recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_, which for now we'll call 'traditional recall'.
It can be most easily explained from an example:

.. note::
   A prediction has to be made if a value will go over a threshold in the near future. We call such an exceedance an incident.
   These incidents have to be predicted before they happen, so someone can act on it. They need some time to act so we want to know
   six to four hours in advance. A prediction will be made every hour on the hour, so at t-6, t-5 and t-4 we expect a positive prediction.
   In our dataset we have 10 incident. Say we predict, for all those 10 incidents, correctly at t-5 (true postive) and incorrectly at t-6 and t-4.
   In this case we have a 'traditional recall' of only 33%.

This 33% is however not representative of how many *incidents* the model actually saw coming in the timespan of t-6 to t-4.
Every incident was predicted to happen correctly at some point in time. To fix this we define **incident recall**.
We *aggregate the positive predictionos per incident* to say if at *any* point during the lead up the incident was correctly predicted.
In our previous (contrived) example the *incident recall* metric would be a 100%, as all incidents were predicted correctly at t-5.

Note that *incident recall >= recall* in every case, traditional recall is a lower bound. Also note that this does not translate to precision.
For precision you care how many of your positive predicitions are correct, regardless if the incident was also correctly predict a moment earlier or later.
We can still use 'traditional' precision.

.. autofunction:: sam.metrics.incident_recall
.. autofunction:: sam.metrics.make_incident_recall_scorer
.. autofunction:: sam.metrics.precision_incident_recall_curve

Mean absolute scaled error
---------------------------
.. autofunction:: sam.metrics.mean_absolute_scaled_error


Train r2
---------------------------
.. autofunction:: sam.metrics.train_r2

Tilted loss
---------------------------
.. autofunction:: sam.metrics.tilted_loss

Keras metrics
-------------
.. autofunction:: sam.metrics.keras_rmse
.. autofunction:: sam.metrics.keras_tilted_loss
.. autofunction:: sam.metrics.keras_joint_mse_tilted_loss
.. autofunction:: sam.metrics.get_keras_forecasting_metrics

Compute quantile ratios
---------------------------
.. autofunction:: sam.metrics.compute_quantile_ratios
