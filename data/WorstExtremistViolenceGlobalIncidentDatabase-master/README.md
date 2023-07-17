# Worst Extremist Violence Global Incident Database
## From 1973 to 2019 (present day)
### Latest update: 14th of July, 2019 NZST
### Number of incidents in dataset: 241
### Database version: 1.16
#### SHA1 hash for version 1.16 of WEVGID.xls: 8a64d8ef21a77fd13a2799d63a9ff8e16878770a
#### The file WEVGID.xls has been signed by felisnigelus@icloud.com


## Introduction

While gathering data about global extremist violence incidents with the intention of analysing the Christchurch Mosques Massacre within a global context, we realised that such data was not readily accessible. It existed but it was spread across many websites and recorded on HTML documents and tables. There was no one-whole set, put together in a standardised manner, in an accessible format (CSV, XLS, or other such), that anyone wishing to do an analysis could download and work with. So we set up to compile it.

Initially we identified two hundred (200) worst incidents of extremist violence, from all over the world, since 1973.

For the purpose of the inclusion of an extremist violence incident in this dataset “*worst*” means one of the following two definitions:
 - the number of direct victims (fatalities plus non-fatally injured) as a ratio of the population of the country in the year of occurrence is 1:1’000,000 or worse – 175 of the 200 incidents identified initially (87.5%) meet this definition;
 - incidents with fewer victims than a 1:1’000,000 ratio but of such infamy that they should be counted amongst the worst; for example, the 2011 Monterrey casino attack (México), the 2007 Samjhauta Express bombings (India), the 2016 Orlando nightclub shooting (USA), the July 2015 Kukawa massacre (Nigeria), or the Nishtar Park bombing (Pakistan).

We will incorporate into the dataset incidents that meet the “worst” criteria as they occur or as we find out about them.

## Contents

The dataset contains the “name” of the incident, the main country (sometimes region) affected, the date(s) when it occurred, the population of the country in the year the incident occurred, the number of fatalities, the number of non-fatally injured, and the number of direct victims (fatalities plus non-fatally injured). All of this data was obtained for each incident from the sources listed below. 

Using this data we have calculated two sets of statistics that can be either used to compare the incidents with each other or simply used to understand a given incident in more detail: 
  - A ratio that shows how many people were unscathed for every fatality (F), non-fatally injured person (I), and direct victim (V). It is calculated by dividing the population (P) of the country in the year the incident occurred by the number of victims and then subtracting 1 (the victim her/himself). This ratio is a static descriptor of the incidents.  The smaller the numerals the worse the incident, for example, an incident with a ratio statistic of 1:10,000 is worse than another incident with a ratio statistic of 1:50,000.
    - Fatalities Ratio: FR = (P÷F) - 1 
    - Non-fatally Injured Ratio: IR = (P÷I) - 1 
    - Direct Victims Ratio: VR = (P÷V) -1

  - A contextual magnitude statistic that locates the impact of a given incident within the frame of reference of the whole set of global extremist violence incidents. It is calculated for each category (fatality, non-fatally injured, direct victims) by dividing the median of all the ratio statistics in the category by the ratio statistic of a given incident. The magnitude statistic is dynamic: as new incidents get added to the dataset and the median changes, it gets recalculated for all incidents.
    - Contextual Magnitude by Fatalities Ratio: MgFR = Median(All FR) ÷ FRx
    - Contextual Magnitude by Non-fatally Injured Ratio: MgIR = Median(All IR) ÷ IRx
    - Contextual Magnitude by Direct Victims Ratio: MgVR = Median(All VR) ÷ VRx

Each incident has its own set of statistics as a function of its number of victims (fatalities, non-fatally injured, direct victims) and the population of the country.

## Methodology

 - The details for each incident are as recorded on the Wikipedia "List-of" articles referenced on the Sources list, except when having been corrected using information obtained from the incident's own Wikipedia article.
 - The dates have been obtained from each incident's own Wikipedia article.
 - When an incident does not indicate a number of non-fatally injured victims, that information is unknown and the field has been left blank.
 - When the number of victims has been uncertain and the victims have been stated as a range (between A and B) the high-end of the range of victims has been the one recorded.
 - The perpetrators are not included in any of the counts.
 - When an incident has happened overnight starting on a date and finishing the following morning, it has been reflected as such on the dates (for example, the 2014 Gamboru Ngala attack).
 - Due consideration has been taken to any likely bias or agenda when gathering information from sources.  Sources have been cross-referenced to reduce bias risk.
 - In all cases of conflicting information, multiple cross-referenced sources have been used.
 - Moving forward, we will be systematically examining compilations of extremist violence incidents in order to find cases that we may have not evaluated and incorporated into the dataset yet.
 - As this database is meant to be an aide for examination and analysis, rather than an exhaustive compilation of every extremist violence incident in history, we have made the design decision to only include incidents from 1973 onwards.

## The Database

[Worst Extremist Violence Global Incident Database](https://github.com/FelisNigelus/WorstExtremistViolenceGlobalIncidentDatabase/blob/master/WEVGID.xls "Version 1.16 14/07/2019 241 records")

 - Each time the file WEVGID.xls is changed a SHA1 hash is computed.
 - Each time the file WEVGID.xls is changed it is [signed](https://github.com/FelisNigelus/WorstExtremistViolenceGlobalIncidentDatabase/blob/master/WEVGID.xls.sig "Signature for Version 1.16 14/07/2019 241 records").

## GitHub
The Worst Extremist Violence Global Incident Database has a repository on [GitHub](https://github.com/FelisNigelus/WorstExtremistViolenceGlobalIncidentDatabase/edit/master/README.md). 

## Sources
[List of battles and other violent events by death toll](https://en.wikipedia.org/wiki/List_of_battles_and_other_violent_events_by_death_toll)

[List of major terrorist incidents](https://en.wikipedia.org/wiki/List_of_major_terrorist_incidents)

[List of events named massacres](https://en.wikipedia.org/wiki/List_of_events_named_massacres)

[List of Islamophobic incidents](https://en.wikipedia.org/wiki/List_of_Islamophobic_incidents)

[Compilation of Antisemitic attacks and incidents](https://en.wikipedia.org/wiki/Category:Antisemitic_attacks_and_incidents)

[Right-wing Terrorism](https://en.wikipedia.org/wiki/Right-wing_terrorism)

[Anti-abortion Violence](https://en.wikipedia.org/wiki/Anti-abortion_violence)

[Significant Acts of Violence against LGBTIQ+](https://en.wikipedia.org/wiki/Significant_acts_of_violence_against_LGBT_people)

[Mass Racial Violence in the United States](https://en.wikipedia.org/wiki/Mass_racial_violence_in_the_United_States)

[Mass Violence & Resistance (MV&R) and Online Encyclopedia of Mass Violence](http://www.sciencespo.fr/mass-violence-war-massacre-resistance/en/homepage)

[Gun Violence Archive (GVA)](https://www.gunviolencearchive.org)

These sources are complemented by each incident's own Wikipedia article; referenced sources in the incident's Wikipedia articles; media of record such as Radio New Zealand, BBC News, Al Jazeera, Mother Jones, CNN, and others; media of record from the countries in which the incidents occurred; plus other authoritative sources, when journalistic sources are inconclusive or contradictory.

We are working through the sources referenced here and are also in the process of identifying other relevant extremist violence compilations and lists.

The population information was obtained from [PopulationPyramid.net](https://www.populationpyramid.net). When a region is not on PopulationPyramid, statistical information is used to obtain its population for the year required.

## Licensing
This project is available under a [Creative Commons Attribution Share Alike 4.0 International](https://github.com/FelisNigelus/WorstExtremistViolenceGlobalIncidentDatabase/blob/master/LICENSE.txt) license.
