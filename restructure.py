import re

with open('docs/appendix_game_projects/intro.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Header and intro
header_match = re.search(r'^(#  D：\n\n> \*\*\*\*：.*?)\n\n## ', content, re.DOTALL)
if not header_match:
    print("Header not found")
header = header_match.group(1)

# 2.  (Milestones)
projects_match = re.search(r'## \n(.*?)(\n## RL |\Z)', content, re.DOTALL)
projects = projects_match.group(1)

# 3. RL 
env_match = re.search(r'## RL \n(.*?)(\n## |\Z)', content, re.DOTALL)
env = env_match.group(1)

# 4. 
learning_match = re.search(r'## \n(.*)', content, re.DOTALL)
learning = learning_match.group(1)

# Constructing new content
new_content = """#  D：

> ****：。、，；，。

## 
""" + learning + """

## 

。 3D ，，，。， 30  RL ，。

###  RL 
""" + env + """
### 
""" + projects

with open('docs/appendix_game_projects/intro.md', 'w', encoding='utf-8') as f:
    f.write(new_content)
