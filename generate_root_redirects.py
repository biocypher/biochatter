import os

# List of redirects from the paper
REDIRECTS = {
    "api-docs/llm_connect": "latest/api-docs/llm_connect",
    "vignettes/custom-bclight-simple": "latest/vignettes/custom-bclight-simple",
    "vignettes/custom-bclight-advanced": "latest/vignettes/custom-bclight-advanced",
    "vignette-custom-bclight-simple": "latest/vignettes/custom-bclight-simple",
    "vignette-custom-bclight-advanced": "latest/vignettes/custom-bclight-advanced",
    "vignettes/custom-decider-use-case": "latest/vignettes/custom-decider-use-case",
    "vignette-kg": "latest/vignettes/kg",
    "vignettes/kg": "latest/vignettes/kg",
    "vignette-rag": "latest/vignettes/rag",
    "vignettes/rag": "latest/vignettes/rag",
    "benchmarking": "latest/features/benchmark",
    "benchmark": "latest/benchmark",
    "benchmark/results": "latest/benchmark/results",
    "chat": "latest/features/chat",
    "rag": "latest/features/rag",
    "features/reflexion-agent": "latest/features/reflexion-agent",
    "reflexion-agent": "latest/features/reflexion-agent",
    "open-llm": "latest/features/open-llm",
    "wasm": "latest/features/wasm",
    "podcast": "latest/features/podcast",
    "llm_connect-reference": "latest/api-docs/llm_connect",
    "vectorstore-reference": "latest/api-docs/vectorstore",
    "kg-reference": "latest/api-docs/kg",
    "api-reference": "latest/api-docs/api-calling-base",
    "reflexion-reference": "latest/api-docs/reflexion",
    "podcast-reference": "latest/api-docs/podcast",
    "benchmark": "latest/benchmark/overview",
    "benchmark-results": "latest/benchmark/results",
    "benchmark-developer": "latest/benchmark/developer"
}

# Output directory for your site
SITE_ROOT = ""

for old, new in REDIRECTS.items():
    # Remove leading/trailing slashes
    old = old.strip("/")
    new = new.strip("/")
    
    # Create directory for the old path
    out_dir = os.path.join(SITE_ROOT, old)
    if os.path.exists(out_dir):
        print(f"WARNING: Skipping redirect for /{old} because {out_dir} already exists.")
        continue
    os.makedirs(out_dir, exist_ok=True)
    
    # Always use an absolute path for the redirect
    absolute_path = '/' + new
    
    # Write index.html with meta-refresh
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv=\"refresh\" content=\"0; url={absolute_path}\" />
    <link rel=\"canonical\" href=\"{absolute_path}\" />
  </head>
  <body>
    <p>Redirecting to <a href=\"{absolute_path}\">{absolute_path}</a>...</p>
  </body>
</html>
""")
    print(f"Created redirect: /{old} -> {absolute_path}")
