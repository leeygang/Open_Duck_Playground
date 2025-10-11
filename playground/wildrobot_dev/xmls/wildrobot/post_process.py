import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def add_option(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)  # Insert at the beginning

    eulerdamp_flag = option.find("flag[@eulerdamp='disable']")
    if eulerdamp_flag is None:
        flag = ET.Element("flag")
        flag.set("eulerdamp", "disable")
        option.append(flag)

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def add_collision_names(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find left_foot body and its foot_btm_front collision geom
    for body in root.findall(".//body[@name='left_foot']"):
        for geom in body.findall("geom[@mesh='foot_btm_front'][@class='collision']"):
            geom.set("name", "left_foot_btm_front")
        for geom in body.findall("geom[@mesh='foot_btm_back'][@class='collision']"):
            geom.set("name", "left_foot_btm_back")

    # Same for right foot
    for body in root.findall(".//body[@name='right_foot']"):
        for geom in body.findall("geom[@mesh='foot_btm_front'][@class='collision']"):
            geom.set("name", "right_foot_btm_front")
        for geom in body.findall("geom[@mesh='foot_btm_back'][@class='collision']"):
            geom.set("name", "right_foot_btm_back")

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)


def add_floating_base_parent(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Find the worldbody element
    worldbody = root.find("worldbody")
    if worldbody is None:
        print("No worldbody found")
        return
    
    if worldbody.find("body[@name='base']") is not None:
        print("found base body, no need to update")
        return

    # Find the waist body
    waist = worldbody.find("body[@name='waist']")
    if waist is None:
        print("No waist body found")
        return
    
    waist_freejoint = waist.find("freejoint[@name='waist_freejoint']")
    if waist_freejoint is None:
        print("no waist_freejoint is found")
    else:
        waist.remove(waist_freejoint)

    # Remove waist from worldbody
    worldbody.remove(waist)
    
    # Create new base body with freejoint
    base_body = ET.Element("body")
    base_body.set("name", "base")
    base_body.set("pos", "0 0 0.5")
    
    # Create freejoint element
    freejoint = ET.Element("freejoint")
    freejoint.set("name", "floating_base")
    # Add freejoint to base body
    base_body.append(freejoint)
    # Add waist as child of base body
    base_body.append(waist)
    # Add base body to worldbody
    worldbody.append(base_body)
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_file)



def main() -> None:
    xml_file = "wildrobot.xml"
    print("start post process...")
    add_collision_names(xml_file)
    add_floating_base_parent(xml_file)
    add_option(xml_file)
    print("Post process completed")


if __name__ == "__main__":
    main()
