Weather data
============

Often, weather features are important predictors. For training the model, historic data might be relevant, but when making predictions, weather forecast can also be useful. However, there is a big difference in the availability of historic weather data vs forecasts, the resolution and frequency of forecasts are for example often lower.
Weather forecasts are never saved, therefore it is not possible to train/validate a model with authentic weather forecasts. In this case, historic data can be used to model both the historic weather features as well as the forecast weather features. However, to make these weather forecast features realistic, they should be modeled according to the availability of the weather forecast data. Weather forecasts might for example have a lower resolution or frequency than historic weather data. If this is the case, the features should resemble this same low resolution or frequency even if they are constructed based on historical data. 

## Historic
Historic weather data is available through KNMI. These can be exported both [hourly](https://projects.knmi.nl/klimatologie/uurgegevens/selectie.cgi) or [daily](http://projects.knmi.nl/klimatologie/daggegevens/selectie.cgi). Depending on the chosen resolution, there are about 30 different variables available, the most interesting are often temperature (eg T, T10N), sunshine (eg SQ, Q), precipitation (eg DR, RH) and wind (eg DD, FH). The measurements are available for around 50 weather stations across the Netherlands, so given a coordinate the closest station should be determined and used.

The KNMI data can be accessed through multiple python packages, for example [knmy](https://pypi.org/project/knmy/) (only export entire years at a time), or [knmi-py](https://pypi.org/project/knmi-py/) (only daily export, not hourly), but is also available in Galileo. The SAM package uses knmy, and can be used with the `sam.data_sources.read_knmi` function.

### Precipitation historic 
Precipitation is often the most important weater feature and is available much more detailed through the [Nationale Regenradar](https://nationaleregenradar.nl/), which is made by Royal HaskoningDHV and Nelen & Schuurmans. It combines weather stations with both Dutch, German and Belgian radar images, resulting in a resolution of 1x1km. The data is available since 2010, with a frequency of 5 minutes. More information can be found [here](https://nationaleregenradar.nl/pdfs/hoofdrapport_NRR_definitief.pdf). 

The data is available through [Lizard](https://rhdhv.lizard.net), but will also be in Galileo in the future (in case of big queries, it might be better to use the raw 'netcdf’s'). In SAM, the lizard data can be used with the `sam.data_sources.read_regenradar` function. This function requires a `user` and `password` to be supplied in the `regenradar` section in the .config file. Loes has such an account. (See .config.example for an example)

## Forecast
KNMI does not (now) have an API to export their forecasts for other weather features besides precipitation. If variables like temperature, sunshine or wind are needed as forecasted features, [OpenWeatherMap](https://openweathermap.org/api) can be used. They have a free 5 day forecast, with a measurement every 3 hours. This forecast is available by city name, geographical coordinates or by zipcode, for any location in the world. A list of all variables that can be exported can be found [here](https://openweathermap.org/forecast5), and includes for example temperature, wind and precipitation. In SAM, this forecast can be obtained with the `sam.data_sources.read_openweathermap` function. This function requires an `apikey` to be supplied in the `openweathermap` section in the .config file. (See .config.example for an example)

### Precipitation forecast
Forecast of precipitation is also available in the Nationale Regenradar. There is a nowcast which forecasts 3 hours with the same frequency (5 minutes) and resolution (1x1 km) as the historic data. The forecast with a bigger horizon is done by KNMI. They make a distinction in midterm prediction till 48 hours (also [HIRLAM](https://data.knmi.nl/datasets/hirlam_p5/0.2?q=hirlam)) and longterm predictions up to 10 days (also [ECMWF](http://projects.knmi.nl/datacentrum/catalogus/catalogus/content/nl-ecm-eps-ts-surf.htm)). The midterm prediction is available with a maximum resolution of 10x10km and a frequency of 6 hours. The longterm prediction with 50x50km and a frequency of 12 hours. Nationale Regenradar directly forwards the KNMI data, without doing any improvements/adaptions. More information can be found [here](https://nationaleregenradar.nl/pdfs/hoofdrapport_NRR_definitief.pdf).

The nowcast is available through [Lizard](https://rhdhv.lizard.net). The KNMI data is not in Lizard, but can be obtained directly through KNMI (see [HIRLAM](http://projects.knmi.nl/datacentrum/catalogus/catalogus/content/nl-nwp-lam-grid-p5.htm) & [ECMWF](http://projects.knmi.nl/datacentrum/catalogus/catalogus/content/nl-ecm-eps-ts-surf.htm)). As of now, this information is not yet available in SAM.


| Type | Data | Source |Frequency | Horizon | Resolution
|---|---|---|---|---|---|
| Historic | Weather | KNMI | 1 hour / 1 day |- | 50 weather stations
| Prediction | Weather | OpenWeatherMap | 3 hours | 5 days | city or coordinate
||
| Historic | Rain | Nationale Regenradar | 5 minutes |- | 1x1 km
| Prediction | Rain | Nationale Regenradar | 3 hours | 5 minutes | 1x1 km
| Prediction | Rain | Nationale Regenradar | 6 hours | 48 hours | 10x10 km
| Prediction | Rain | KNMI | 12 hours | 10 days | 50x50 km