# Greenhouse-Growing-Challenge
- For the machine learning challenge, teams will get access to a simple greenhouse climate and lettuce production model (simple simulator) during the preparation phase.

- The simple simulator consists of a given set of outside climate conditions, a given greenhouse type and given greenhouse actuators (ventilation, heating, lighting, screening). It needs to be provided with a series of climate setpoints (ventilation strategy, heating strategy, lighting strategy, screening strategy per timestep) as inputs. The input climate setpoints will activate the available actuators, which will control the inside greenhouse climate. The realised inside climate parameters will be provided as a feed back value. Crop management consists of defining plant density (number of plants m-2) over time. Since the crop growth in the simulator is determined by the realised greenhouse climate, also the crop growth parameters (fresh weight, height, diameter) over time will be provided as output.

- The climate control strategy will determine the use of resources, mainly energy (for heating, for electricity for artificial light) and therefore creates costs.

- Fresh weight, height and diameter of the average lettuce plant are provided as the main output. These determine product price and therefore create income.

- Teams will have to develop machine learning algorithms to feed the simple simulator with the optimised control parameters in order to maximise net profit.

- During the preparation phase teams can interact with the simple simulator for algorithm development. During the Online Challenge this algorithm should be suitable to control the growth of a virtual crop in a virtual greenhouse under changed conditions (e.g. other weather conditions, different greenhouse type, different lettuce type) and limited time constraints.

- There will be different versions of the simulators (A-D) with slightly different simulation parameters (e.g. other weather conditions, different greenhouse type, different lettuce type):
  - Simulator A will be avaible during the preparation phase of the Online Challenge form 1 June 0:00 CET to 11 July 23:00 CET, with limited access (typically 1000 times per day), to train the algorithms of the teams.
  - Simulator B will be available once every day from 12:00-13:00 CET, to test the trained model, with limited access (typically 200 times per day). A public ranking board will be generated according to the net profit on simulator B. On July 11th 2021 23:00 CET Simulator A and B will be closed.
  - On July 12th 00:00 CET, Simulator C will be available for a period of 24 h, with limited access (typically 1000 times), to re-train the model.
  - On July 13th 12:00 CET a private ranking board will be opend and simulator D will be provided to the teams with limited access (typically 200 times). Teams must submit the optimised control parameters in this new simulator version based on their developed algorithms in order to maximise net profit. After July 13th 13:00 the submitting will be rejected. Teams can only see their own scores in the private board before the announcement of the final results realised in simulator D on July 14th.