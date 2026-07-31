The project aims to construct hierarchical stylized samples from non-parallel style-transfer data. Instead of directly prompting large language models (LLMs) to generate predefined style-intensity levels, the framework discovers local stylistic weaknesses, generates candidate repairs through weakness combinations, and validates progressive hierarchical levels using boundary classifiers.  

##Project Overview  

The main workflow includes:  

Weakness discovery: identifying local stylistic weaknesses between source texts and target-style references.  
Weakness-guided repair: generating candidate stylized texts by selectively repairing discovered weaknesses.  
Boundary validation: using binary classifiers to determine whether repaired candidates cross the current style boundary.  
Hierarchical construction: iteratively constructing progressive stylized samples across multiple levels.  
Task IDs  

The implemented tasks are indexed as follows:  

task 2: positive style  
task 3: negative style   
task 4: formal style  
task 5: informal style  
task 6: neutral style  
task 7: toxic style  

##How to Run  

The main entry point is:  

python main.py  

The construction process starts from level-0. Subsequent levels, such as level-1, level-2, and level-3, should be specified manually according to the experimental setting.  

##Cached LLM Outputs  

The LLM request results for all tasks (task 2–task 7) have already been saved in this repository. Under the default settings, the cached request results are directly used, and users do not need to send new API requests to reproduce the reported hierarchical construction results.  

If users want to re-run the LLM request stage, they need to configure the corresponding API keys in main.py or adapt the code to load API keys from local environment variables.  

##Output Directory  

The final hierarchical stylized samples are saved in:  

distill_result/test_result  

##Reproducibility Note  

To support reproducibility, this repository provides the cached LLM-generated intermediate results, final hierarchical construction results, and task-level outputs used in the experiments. Users can reproduce the main construction process without re-querying the LLM APIs under the default configuration.  
