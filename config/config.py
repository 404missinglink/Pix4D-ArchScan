import os

video_folder_path = os.path.join("data", "video")

prompt = (
                    '''
Provide a detailed analysis of this frame based on the following key areas relevant to structural and construction surveying. The response should be in plain text format with each key area as a separate section. Ensure the information is concise, direct, and easily processed by a larger language model. Additionally, identify and describe the type of project (e.g., building, bridge, road) present in the frame to provide more context.

### Key Areas:

- **General Structural Condition**: 
  - Foundation
  - Walls
  - Roof

- **External Features**: 
  - Façade & Cladding
  - Windows and Doors
  - Drainage and Gutters

- **Internal Condition**: 
  - Floors and Ceilings
  - Walls
  - Electrical and Plumbing

- **Signs of Water Damage or Moisture**: 
  - Stains or Discoloration
  - Basement & Foundation

- **HVAC Systems** (if visible)

- **Safety Features**: 
  - Fire Exits
  - Handrails and Guardrails

- **Landscaping & Surroundings**: 
  - Site Drainage
  - Paths and Roads
  - Tree Proximity

- **Construction Progress** (if an active project): 
  - Consistency with Plans
  - Material Usage
  - Workmanship

- **Temporary Supports & Site Safety** (if under construction): 
  - Scaffolding
  - Temporary Structures

- **Building Services** (if visible): 
  - Mechanical & Electrical Installations
  - Elevators & Staircases

- **Project Type**: Identify and describe the type of project (e.g., building, bridge, road).
'''
                )

TEXT_MODEL = "mistral-large-latest"
PIXTRAL_API_URL = "http://127.0.0.1:8888/describe_image"


RATE_LIMIT_SECONDS = 1.0
TRIM_START_FRAMES = 30
TRIM_END_FRAMES = 30