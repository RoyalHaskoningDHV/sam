# Nereda

## Decisions

* We focus on two Nereda plants: Garmerwolde and Epe [Kim, Bart]

## Data Discoveries (tm)

### Expected values
Information provided by Bart.
| **Variable** | **Values**
| NH4 | 0 - 20
| NO3 |  0 - 20
| PO4: | 0 - 10
| Turbidity | 10-30, but can be significantly higher. More than 100 is improbable

- Looking at only GMW we see a decrease in how often there is more than 30 minutes between batches, from 2015 on we see between 0 and 10 monthly occurrences (mostly less than 5)
- GMW seems to have a little longer batches than EPE, the a threshold of 420 shows a lot less 'too long' batches than 370.

### Data to ignore [should be verified]

- EPE: 2011-12-31 till 2012-03-19 often have more than 30 minutes between the batches in a reactor
- EPE **reactor 2** has often has more than 30 minutes between the batches in 2016-01-20 till more ore less 2016-03-01
- From around 2013-06 till 2014-06 a lot of batches in EPE took more than 420 minutes.
- Some time around 2013-08 (need to check) GMW has a lot of batches longer than 420 minutes.

### Known issues

- The API returns duplicate entries sometimes (timestamps might be 2 hours off)
        - EPE 1 on `2018-02-24 21:12:52`, completely duplicate [discovered 18-10-2018]
        - GMW 2 on `12141 2017-07-18 13:23:46` duplicate finish date but different values [discovered 18-10-2018]
- Sometimes data is missing, if //all// reactors are missing this is a data collection error and should be ignored. If one is missing this is interesting point [Kim, Bart]
- Sometimes batches 'hang' in their measurements and values of batches are combined. Examples of this:
  - 09-11-2017 20:24:20
  - 09-11-2017 22:55:37
  - These timestamp are a little off because of some timezone messup, in the data it is correct but you should be able to find them
   {F5677}

## General information

- Reactors aren't linked 'biologically', ie: a problem in reactor one does not directly transfer to other reactors. **However:** The cause of a problem (something in the water) can of course manifest in the other reactors. Also note that when  a reactor is not working properly the other reactors might have more to do and will thus also be influenced.
- GMW in 2014 ran 'normally' in the sense that they did not try to use it as much as possible. Data in the years after is running the plant at maximum to use the old (more expensive) system less. [Bart]
- Rainfall influences the process [Kim, Bart]
- We can take a look at the worldcup dates, should have peaks in influent

### Weather locations

* Epe --> Heino?: 278
* Garmerwolde --> Eelde: 280
