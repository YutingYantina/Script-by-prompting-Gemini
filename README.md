script folder 
contains generated scripts

generate_script.py
Prompt for a single herb, contain 4 samples and output 2 types of scripts (medical use, producing location, appearance, which medicine contains the herb) and (ancient or original story), output as a txt

generate_all_herbs.py
Use prompt with vertexai. 
Get 153 herbs in total after cleanning (some are serve for ai to generate while others is too rare)
Meet problem:
_MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated with:
	status = StatusCode.RESOURCE_EXHAUSTED
	details = "Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute_per_project_per_base_model with base model: gemini-1.5-flash. Please submit a quota increase request. https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai."

herb_name
About 171 herbs
