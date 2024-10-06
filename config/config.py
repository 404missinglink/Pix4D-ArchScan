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


prompt2 = (
    """
You are an expert in analyzing under-construction structural and civil engineering projects. 
I am providing you with a frame from a drone or aerial video of a construction site, which could involve buildings, roads, bridges, or other infrastructure.
Your task is to comprehensively analyze the visible aspects of the current frame, focusing on structural integrity, construction progress, and safety features. 
Provide detailed observations in the following key areas:

### Key Areas to Focus On:

1. **General Structural Condition:**  
   Evaluate the main structure, whether its a building or road project.  
   - **Foundation (for Buildings or Bridges):** Look for any visible signs of cracks, settlement, or incomplete work.  
   - **Walls (for Buildings):** Check for structural integrity, visible cracks, or misalignment.  
   - **Roof (for Buildings):** Assess if there are any incomplete sections, sagging, or poor material installation.  

2. **External Features (for Buildings):**  
   Focus on external elements of any visible building structure.  
   - **Façade & Cladding:** Evaluate the uniformity and installation quality of external cladding or façade materials.  
   - **Windows and Doors:** Identify missing, misaligned, or incomplete installations.  
   - **Drainage and Gutters:** Inspect whether drainage systems are visible, installed correctly, and operational.  

3. **Road Alignment & Geometry (for Roads):**  
   For road construction, analyze the overall alignment and design of the road.  
   - **Curvature and Gradient:** Assess the horizontal and vertical alignment to ensure it follows proper design standards.  
   - **Lane Markings and Signage:** Identify if lane markings, signs, or signals are visible and placed correctly.  
   - **Intersections and Turnouts:** Look for progress in intersections, turnouts, or any other road junctions.  

4. **Signs of Water Damage or Moisture:**  
   Identify any water-related issues that may affect buildings or roads.  
   - **Stains or Discoloration (for Buildings):** Look for signs of water seepage or moisture accumulation on walls or roofs.  
   - **Basement & Foundation (for Buildings):** Examine whether there are any visible cracks or damp areas near the foundation.  

5. **Safety Features:**  
   Evaluate the safety measures implemented on the site for both construction workers and future occupants.  
   - **Handrails and Guardrails (for Buildings & Bridges):** Check for the presence of handrails or guardrails around elevated areas.  
   - **Traffic Barriers (for Roads):** Ensure that traffic barriers or cones are in place near work zones or sharp curves to protect road workers and vehicles.  

6. **Landscaping & Surroundings:**  
   Consider the interaction between the site and its surrounding environment, particularly in terms of safety and drainage.  
   - **Paths and Roads (for Building Sites):** Look for access routes for vehicles and pedestrians on the construction site.  
   - **Tree Proximity (for Buildings & Roads):** Assess the impact of nearby trees on the project, considering risks like root encroachment or falling branches.  

7. **Construction Progress:**  
   Evaluate the progress of construction compared to expected milestones, whether for buildings, roads, or other infrastructure.  
   - **Consistency with Plans (for Buildings & Roads):** Ensure that the project aligns with construction plans and that the structural components are in place.  
   - **Material Usage:** Look for signs of proper material usage or evidence of waste and mismanagement.  
   - **Workmanship:** Assess the overall quality of work, including finishing details like smoothness of surfaces, precise alignment, and proper material application.  


8. **Bridge Construction:**  
    If the frame shows bridge construction, evaluate the main components of the bridge.  
    - **Piers and Abutments:** Assess their stability and alignment.  
    - **Bridge Deck:** Inspect for progress in the deck construction and check for defects.  
    - **Suspension and Cables (for Suspended Bridges):** Check if the suspension system or cables are properly installed and secure.  

9. **Project Type:**  
    Based on visible cues, identify whether the project is a building, road, bridge, or other civil infrastructure. Provide a brief description of the project and its current construction stage. 


"""
)


TEXT_MODEL = "mistral-large-latest"
PIXTRAL_API_URL = "http://127.0.0.1:8888/describe_image"


RATE_LIMIT_SECONDS = 1.0
TRIM_START_FRAMES = 30
TRIM_END_FRAMES = 30