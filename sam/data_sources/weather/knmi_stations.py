import pandas as pd
from io import StringIO


"""
Location of all weather stations available in knmy, with coordinates
List of all weather stations can be found here:
http://projects.knmi.nl/datacentrum/catalogus/catalogus/content/nl-obs-surf-stationslijst.htm
"""

# No csv file to avoid problems with relative paths within a package
csv = """
number,name,latitude,longitude
209,IJmond,52.465,4.518
210,Valkenburg Zh,52.171,4.430
215,Voorschoten,52.141,4.437
225,IJmuiden,52.463,4.555
235,De Kooy,52.928,4.781
240,Schiphol,52.318,4.790
242,Vlieland,53.241,4.921
248,Wijdenes,52.634,5.174
249,Berkhout,52.644,4.979
251,Hoorn Terschelling,53.392,5.346
257,Wijk aan Zee,52.506,4.603
258,Houtribdijk,52.649,5.401
260,De Bilt,52.100,5.180
265,Soesterberg,52.130,5.274
267,Stavoren,52.898,5.384
269,Lelystad,52.458,5.520
270,Leeuwarden,53.224,5.752
273,Marknesse,52.703,5.888
275,Deelen,52.056,5.873
277,Lauwersoog,53.413,6.200
278,Heino,52.435,6.259
279,Hoogeveen,52.750,6.574
280,Eelde,53.125,6.585
283,Hupsel,52.069,6.657
285,Huibertgat,53.575,6.399
286,Nieuw Beerta,53.196,7.150
290,Twenthe,52.274,6.891
308,Cadzand,51.381,3.379
310,Vlissingen,51.442,3.596
311,Hoofdplaat,51.379,3.672
312,Oosterschelde,51.768,3.622
313,Vlakte van De Raan,51.505,3.242
315,Hansweert,51.447,3.998
316,Schaar,51.657,3.694
319,Westdorpe,51.226,3.861
323,Wilhelminadorp,51.527,3.884
324,Stavenisse,51.596,4.006
330,Hoek van Holland,51.992,4.122
331,Tholen,51.480,4.193
340,Woensdrecht,51.449,4.342
343,Rotterdam Geulhaven,51.893,4.313
344,Rotterdam,51.962,4.447
348,Cabauw Mast,51.970,4.926
350,Gilze-Rijen,51.566,4.936
356,Herwijnen,51.859,5.146
370,Eindhoven,51.451,5.377
375,Volkel,51.659,5.707
377,Ell,51.198,5.763
380,Maastricht,50.906,5.762
391,Arcen,51.498,6.197
"""

knmi_stations = pd.read_csv(StringIO(csv))
