# style_trans

This project is used to iteratively construct hierarchical stylized samples from LLMs using non-parallel text data.
The main function is executed by running main.py.
Tasks 2–7 represent positive, negative, formal, informal, neutral, and toxic, respectively.
To run, you need to start from level-0, then manually set level-1, level-2, and level-3.
The LLM request results for all tasks (2–7) have already been saved in the project. Under default settings, there is no need to re-request them.
To make requests, you need to set the API key for each LLM in the main function of main.py.
The final hierarchical results are saved in distill_result/test_result.
