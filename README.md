The project aims to construct hierarchical stylized samples from non-parallel style-transfer data. Instead of directly prompting large language models (LLMs) to generate predefined style-intensity levels, the framework discovers local stylistic weaknesses, generates candidate repairs through weakness combinations, and validates progressive hierarchical levels using boundary classifiers.  

## Project Overview

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

## How to Run  

Before running the project, please check the following settings in main.py.  

First, this project uses LLM APIs during the weakness discovery and repair generation stages. Therefore, the corresponding API keys should be configured in main.py before sending new LLM requests. For security reasons, users are advised to use their own local API keys and avoid committing private keys to the repository.  

Second, this project uses an SBERT model and a Llama-3.2-3B model for semantic representation and classifier-based validation. These models are not included in the repository. Users need to download them separately and modify the corresponding model paths in main.py.  

Third, because the framework needs to process structured outputs returned by LLMs, several functions are implemented to check whether the LLM outputs follow the required format. If an abnormal output format is detected, the program may stop, and the user needs to manually correct or reprocess the problematic output. Since the intermediate files provided in this repository have already been processed and corrected, running the project under the default settings does not require additional format checking.  

Finally, this project is designed to iteratively construct hierarchical stylized samples. Each execution of main.py performs one iteration of the construction process. To conduct multiple rounds of iteration, users need to manually specify the current level, such as level-0, level-1, level-2, and so on, in main.py. The iteration stopping criteria include the boundary classifier accuracy and the pseudo-positive sample retention rate, both of which can be modified in main.py.    

## Cached LLM Outputs  

The LLM request results for all tasks (task 2–task 7) have already been saved in this repository. Under the default settings, the cached request results are directly used, and users do not need to send new API requests to reproduce the reported hierarchical construction results.  

If users want to re-run the LLM request stage, they need to configure the corresponding API keys in main.py or adapt the code to load API keys from local environment variables.  

## Output Directory  

The final hierarchical stylized samples are saved in:  

distill_result/test_result  

## Reproducibility Note  

To support reproducibility, this repository provides the cached LLM-generated intermediate results, final hierarchical construction results, and task-level outputs used in the experiments. Users can reproduce the main construction process without re-querying the LLM APIs under the default configuration.  
