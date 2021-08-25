### PHE Covid-19 Data Gathering

This file provides the links on where the data for the analytical team has been sourced and gathered.

Main data source has been the Public Health England's data service:

PHE Data download - https://coronavirus.data.gov.uk/details/download

The data is being provided at different granularities so we need to do a bit of matching and mapping. While data on cases, deaths and vaccinations are available at Upper Tier Local Authority (UTLA) level, data on hospitalizations are available at NHS Trust Level. Since we want to work on data at UTLA level, we need to map the hospitalisations to the individual UTLAs. However, there is not a direct mapping between UTLAs to NHS Trusts as NHS Trusts can get patients from various UTLAs within their vicinity/catchment. Therefore, we used a library by LSHTM folks (https://epiforecasts.io/covid19.nhs.data/reference/index.html) that probabilistically maps UTLAs with NHS Trusts. This enables us to have a joint dataset at UTLA level that has all the combined information.


### At UTLA level 

The core 

![image-20210730180500664](https://tva1.sinaimg.cn/large/008i3skNgy1gt75n4qlhjj31p00u0dkn.jpg)

https://api.coronavirus.data.gov.uk/v2/data?areaType=utla&metric=cumVaccinationCompleteCoverageByVaccinationDatePercentage&metric=newCasesByPublishDate&metric=newDeaths28DaysByDeathDate&metric=cumVaccinesGivenByPublishDate&format=csv

**Downloaded:** 04.08.2021, 23:46

### Hospital Admissions at NHS Trust Level from PHE

![image-20210805142636237](https://tva1.sinaimg.cn/large/008i3skNgy1gt75n8vmogj31ry0tq77u.jpg)

https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsTrust&metric=newAdmissions&metric=covidOccupiedMVBeds&format=csv

#### Hospital Admissions at UTLA level

The above data is mapped to UTLAs using the mapping (NHS Trust to UTLA) provided by this library by LSTHM folks:
https://epiforecasts.io/covid19.nhs.data/reference/index.html

The Jupyter Notebook in the folder takes care of this mapping.


# Meta-data

Now that we have the COVID-19 related data at UTLA level, next operation is to link these data with other health indicators and wider determinants of health as provided by the Public Health Outcomes Framework.

### Public Health Outcomes Framework

https://fingertips.phe.org.uk/profile/public-health-outcomes-framework

Downloaded from here: https://fingertips.phe.org.uk/profile/public-health-outcomes-framework/data#page/9/gid/1000049/pat/6/par/E12000001/ati/402/are/E06000047/iid/90362/age/1/sex/1/cid/4/tbm/1

### Population data that is at UTLA level

To be able to normalise counts/numbers by population, here is the ONS resource:
- https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/localauthoritiesinenglandtable2
- PHE C-19 Dashboard now also hosts a age-braket broken down population counts - https://coronavirus.data.gov.uk/details/download
