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
209,IJmond,52.47,4.52
210,Valkenburg Zh,52.18,4.42
215,Voorschoten,52.13,4.43
225,IJmuiden,52.47,4.57
229,Texelhors,53,4.45
235,De Kooy,52.93,4.78
240,Schiphol,52.32,4.87
242,Vlieland,53.23,4.92
248,Wijdenes,52.63,5.17
249,Berkhout,52.65,4.98
251,Hoorn Terschelling,53.38,5.35
257,Wijk aan Zee,52.3,4.36
258,Houtribdijk,52.64,5.39
260,De Bilt,52.1,5.18
265,Soesterberg,52.13,5.28
267,Stavoren,52.9,5.38
269,Lelystad,52.45,5.52
270,Leeuwarden,53.22,5.52
273,Marknesse,52.7,5.88
275,Deelen,52.05,5.87
277,Lauwersoog,53.42,6.2
278,Heino,52.43,6.27
279,Hoogeveen,52.75,6.57
280,Eelde,53.12,6.58
283,Hupsel,52.07,6.65
285,Huibertgat,53.57,6.4
286,Nieuw Beerta,53.2,7.15
290,Twenthe,52.27,6.88
308,Cadzand,51.38,3.38
310,Vlissingen,51.45,3.6
311,Hoofdplaat,51.38,3.67
312,Oosterschelde,51.77,3.62
313,Vlakte van De Raan,51.5,3.25
315,Hansweert,51.45,4
316,Schaar,51.65,3.68
319,Westdorpe,51.22,3.87
323,Wilhelminadorp,51.53,3.88
324,Stavenisse,51.6,4
330,Hoek van Holland,51.98,4.12
331,Tholen,51.52,4.13
340,Woensdrecht,51.54,4.35
343,Rotterdam Geulhaven,51.88,4.32
344,Rotterdam,51.97,4.45
348,Cabauw Mast,51.97,4.92
350,Gilze-Rijen,51.57,4.93
356,Herwijnen,51.85,5.15
370,Eindhoven,51.45,5.38
375,Volkel,51.65,5.7
377,Ell,51.2,5.77
380,Maastricht,50.9,5.77
391,Arcen,51.5,6.2
"""

knmy_stations = pd.read_csv(StringIO(csv))
