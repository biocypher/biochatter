Our design philosophy is based on open-source principles and governed by the
desire to keep up with the rapid pace of LLM development. We adopt an agile
style of developing modular components, aiming for quick iteration on
proof-of-concept implementations. The rapid introduction of new models and
functionalities (e.g., tool binding) necessitates a workflow that can quickly
adapt to these changes.

This often involves the need of frequent refactorings of individual modules of
the codebase, making modularity and separation of concerns a vital concern.
Further, in order to facilitate collaboration and onboarding of new
contributors, the codebase needs to be easily understandable and
well-documented.

## Feature Selectiveness

For the average research initiative, even when factoring in community
contributions, it is not feasible to compete with the resources of large
companies, which are numerous in the current LLM ecosystem. As a result, it
often is not sensible to work on solutions that are subject to high company
interests. For instance, early BioChatter developments included a workflow for
automating podcast-like summaries of scientific papers; we discontinued this
effort in light of later developments by Google to provide this same
functionality inside their NotebookLM platform. We aim to continue to be mindful
of this issue when deciding on which features to implement, which requires
constant monitoring of the broader LLM ecosystem.
