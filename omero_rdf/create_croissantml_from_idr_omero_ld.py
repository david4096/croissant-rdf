import json
import mlcroissant as mlc
import re

omero_jsonld = "example_data\omero_project_801.jsonld"
croissant_path = omero_jsonld.replace(".jsonld", "_croissant.json")


def fetch_map_annotations(omero_jsonld):
    annotations = [a for a in omero_jsonld["Annotations"] if a["@type"].endswith("MapAnnotation")]
    annotations_dict = {}
    for a in annotations:
        value = a["Value"]
        for kv_tuples in value:
            annotations_dict[kv_tuples[0]] = kv_tuples[1]
    return annotations_dict

def filter_annotations(omero_jsonld_dict, filter_keys):
    filter_fn = lambda k: re.search("|".join(filter_keys),k[0], re.IGNORECASE)
    filtered_entries = list(filter(filter_fn, omero_jsonld_dict.items()))
    return filtered_entries
    
with open(omero_jsonld) as f:
    omero_jsonld_dict = json.load(f)

annotations_dict = fetch_map_annotations(omero_jsonld_dict)

distribution = [
    # gpt-3 is hosted on a GitHub repository:
    mlc.FileObject(
        id="project-annotation",
        name="project-annotation",
        description="CSV annotation for the images, including biological experiment details.",
        content_url=annotations_dict["Annotation File"].split(" ")[1],
        sha256="main",
        encoding_format="application/json"
    ),
    # Within that repository, a FileSet lists all JSONL files:
    mlc.FileSet(
        id="image-example-planes",
        name="image-example-planes",
        description="Example data planes, template. Use OMERO CLI and Images ID to download the full image.",
        encoding_format="application/json",
        includes=omero_jsonld_dict["images_data"],
    ),
    mlc.FileSet(
        id="image-metadata",
        name="image-metadata",
        description="Metadata of the images in the project.",
        encoding_format="application/json",
        includes=omero_jsonld_dict["images_metadata"],
    ),
]

filtered_authors = filter_annotations(annotations_dict, ["author", "creator"])
authors = filtered_authors[0][1] if len(filtered_authors) > 0 else "Unknown"
authors = [{"@type": "Person", "name": a.strip()} for a in authors.split(",")]

filtered_license = filter_annotations(annotations_dict, ["license"])
license = filtered_license[0][1] if len(filtered_license) > 0 else "Unknown"

filtered_citation = filter_annotations(annotations_dict, ["cit", "doi"])
citation = filtered_citation[0][1] if len(filtered_citation) > 0 else "Unknown"

metadata = mlc.Metadata(
    name = omero_jsonld_dict["Name"],
    description= omero_jsonld_dict["Description"], 
    url = omero_jsonld_dict["@id"],
    distribution=distribution,
    creators=authors,
    license=license,
    cite_as=citation
) # TODO add creators, license from the annotations

#print(metadata.issues.report())


with open(croissant_path, "w") as f:
  content = metadata.to_json()
  content = json.dumps(content, indent=2)
  print(content)
  f.write(content)
  f.write("\n")  # Terminate file with newline