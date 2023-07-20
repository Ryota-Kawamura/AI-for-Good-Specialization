# Datasheet: *Disaster response data* Lab 3

Author: DeepLearning.AI (DLAI)

Files:
	disaster_response_training.csv
	disaster_response_validation.csv
	disaster_response_test.csv
	haiti_df_processed.pkl


## Motivation

This dataset contains 25,000 messages drawn from events including an earthquake in Haiti in 2010, floods in Pakistan in 2010, Hurricane Sandy in the USA in 2012, and news articles spanning a large number of years and 100s of different disasters. The data has been encoded with 38 different categories related to disaster response and has been stripped of messages with sensitive information.

This dataset is downloaded and used as is from the following website: https://github.com/rmunro/disaster_response_messages
A very comprehensive datasheet is also found on the above website.

The data is collected and compiled by Robert Munro, who worked and led data teams on these disasters and wrote a PhD thesis about it. See the citation below:

Robert Munro. 2012. Processing short message communications in low-resource languages. [PhD dissertation, Stanford University]. Stanford Digital Repository. Retrieved from https://purl.stanford.edu/cg721hb0673

In addition to the original files, there is the haiti_df_processed.pkl file, which is a pickled Pandas dataframe containing only the Haiti data compiled from all three .csv files above and includes some additional columns. The file was created in the previous "design phase 1" notebook.

## Composition

There are 38 categories, plus the messages themselves and (for Haiti only) the dates of the messages.

The categories are hierarchical, with sub-categories for `aid_related`, `infrastructure_related`, and `weather_related`.

* id: Unique ID number for the messages. The IDs are in (roughly) the order that the messages were written.
* split: `training`, `validation` or `test` which should correlate with the files the data is shared in.
* message: the English message
* original: in the case of non-English messages in Haiti and Pakistan, the original message before translation
* genre: `direct` message or `news` headline
* related: `0`, `1` or `2`, whether the message is related to a disaster (`1` == yes, `0' == no, `2` == unsure)
* PII:  `0` or `1`, whether the message is related to a disaster (all `0` in this public release)
* request: `0` or `1`, whether the message is a request for aid
* offer: `0` or `1`, whether the message is offering help
* aid_related: `0` or `1`, whether the message is related to aid

    * medical_help: `0` or `1`, whether the message is about medical help
    * medical_products: `0` or `1`, whether the message is about medical products
    * search_and_rescue: `0` or `1`, whether the message is about search and rescue
    * security: `0` or `1`, whether the message is about personal security
    * military: `0` or `1`, whether the message is about military actions
    * child_alone: `0` or `1`, whether the message is about a child/children who are without adult care (all `0` in this public release)
    * water: `0` or `1`, whether the message is about drinking water
    * food: `0` or `1`, whether the message is about food
    * shelter: `0` or `1`, whether the message is about shelter
    * clothing: `0` or `1`, whether the message is about clothing
    * money: `0` or `1`, whether the message is about money
    * missing_people: `0` or `1`, whether the message is about missing people
    * refugees: `0` or `1`, whether the message is about refugees or internally displaced people
    * death: `0` or `1`, whether the message is about death
    * other_aid: `0` or `1`, whether the message is about another aid-related topic
    
* infrastructure_related: `0` or `1`, whether the message is about infrastructure-related issues

    * transport: `0` or `1`, whether the message is about transport like buses, trains, planes, boats, taxis, bicycles, etc. and interuptions to transport like blocked roads or missing bridges.
    * buildings: `0` or `1`, whether the message is related to buildings: unstable, collapsed, inundated, usable as shelters, etc. 
    * electricity: `0` or `1`, whether the message is related to power infrastructure, including public utilities and private generators
    * tools: `0` or `1`, whether the message is about tools related to disaster prevention and response
    * hospitals: `0` or `1`, whether the message is related to infrastructure for medical care, including hospitals and makeshift clinics 
    * shops: `0` or `1`, whether the message is related to shops, markets, and other places of commerce, real or online
    * aid_centers: `0` or `1`, whether the message is related to aid_centers
    * other_infrastructure: `0` or `1`, whether the message is related to other types of disaster-related infrastructure
    
* weather_related: whether the message is weather-related

    * floods: `0` or `1`, whether the message is related to flooding
    * storm: `0` or `1`, whether the message is related to storms, including hurricanes, tornadoes and snow-storms
    * fire: `0` or `1`, whether the message is related to fire, including house fires and bush/forest fires
    * earthquake: `0` or `1`, whether the message is related to earthquakes
    * cold: `0` or `1`, whether the message is related to dangers from cold weather
    * other_weather: `0` or `1`, whether the message is related to other weather events
    
* direct_report: `0` or `1`, whether the message is a direct report from someone experiencing/witnessing the disaster or if they are reporting second/third hand
* event: which event, `haiti_earthquake`, `pakistan_floods`, `usa_sandy`, or `NULL` for news
* actionable_haiti: `0`, `1` or `NULL`, was this message considered something that could be responded to at the time? (Haiti only)
* date_haiti: `(YYY-MM-DD)` or `null', the date the message was sent (Haiti only)

The categories are are motivated by different aspects of disaster reponse work:

The "related" category refers to any disaster preparation or response that an aid agency might respond to. This will not include memorials about past disasters or accidents/crimes that are independent of a disaster, but it will include individual Search & Rescue operations. This is the same definition of "disaster related" used in my news headlines dataset ([https://github.com/rmunro/pytorch_active_learning](https://github.com/rmunro/pytorch_active_learning)), which can be used in combination with the dataset here for the "related" category. Like the description of that dataset says, the definition of "disaster related" can perpetuate biases, especially in the case of news reporting where a person's religion and ethnicity might determine whether a violent event is seen as an individual crime or part of a disaster like a "war on terror".

The "offer" and "request" categories represent the common task of resource-matching in a disaster to help people within the crisis-affected communities find each other and coordinate their own rescue. Most disasters responders are from among the crisis-affected communities, not professional responders, so this use case is very important. 

The "aid_related" subcategories represent the response offered by different types of aid organizations and could support the task of filtering information relevant to a certain organization. That explains why the ontology might seem to be overly fine-grained in some places. For example, it will often be different organizations who provide medical help and who provide medical products, so these are separate categories. Similarly, different organizations will often respond to food needs and water needs, because water needs are often more time critical but in some cases can be solved locally with treatment.

The "infrastructure_related" subcategories relate to situation awareness to predict where situations like a lack of power could lead to worsening situations or where the logistics to deliver aid could be complicated by transportation closures.

The "weather_related" subcategories relate to different types of natural disasters. 

The "direct_report" category can help remote responders identify eye-witnesses to disasters, which can be important people to communicate with when there is uncertain information about how the disaster is unfolding due to rumours or deliberate misinformation.

The "actionable_haiti" category is a specific definition of what could be responded to by international aid organizations during that disaster. These are the actual labels that we gave these messages at the time. In any disaster the definition of "actionable" will often be context-specific and changing. In the Haiti data this was initially Search and Rescue and medical emergencies, and then any individual request for drinking water, and later requests by groups of people who need food (but not individuals unless children). Any model built to automatically identify actionable information in a disaster will therefore need to be similarly time-sensitive or otherwise adaptive, which is a difficult task for machine learning.


The data is split into training, validation, and test datasets randomly.
