# @Copyright: CEA-LIST/DIASI/SIALV/LVA (2023)
# @Author: CEA-LIST/DIASI/SIALV/LVA <pixano@cea.fr>
# @License: CECILL-C
#
# This software is a collaborative computer program whose purpose is to
# generate and explore labeled data for computer vision applications.
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
#
# http://www.cecill.info

import glob
import pandas as pd
from collections.abc import Iterator
from pathlib import Path

from pixano.core import Image
from pixano.data.importers.importer import Importer
from pixano.utils import image_to_thumbnail, natural_key


class MnistImporter(Importer):
    """Importer class for Mnist

    Attributes:
        info (DatasetInfo): Dataset information
    """

    def __init__(
        self,
        name: str,
        description: str,
        splits: list[str],
    ):
        """Initialize Mnist Importer

        Args:
            name (str): Dataset name
            description (str): Dataset description
            splits (list[str]): Dataset splits
        """

        tables = {
            "main": [
                {
                    "name": "db",
                    "fields": {
                        "id": "str",
                        "views": "[str]",
                        "split": "str",
                    },
                }
            ],
            "media": [
                {
                    "name": "image",
                    "fields": {
                        "id": "str",
                        "image": "image",
                    },
                }
            ]#,
            # "objects": [
            #     {
            #         "name": "objects",
            #         "fields": {
            #             "category_id": "int",
            #             "category_name": "str",
            #         },
            #         "source": "Ground Truth",
            #     }
            # ],
        }

        # Initialize Importer
        super().__init__(name, description, tables, splits)

    def import_rows(
        self,
        input_dirs: dict[str, Path],
        portable: bool = False,
    ) -> Iterator:
        """Process dataset rows for import

        Args:
            input_dirs (dict[str, Path]): Input directories
            portable (bool, optional): True to move or download media files inside dataset. Defaults to False.

        Yields:
            Iterator: Processed rows
        """

        # iterate on splits
        for split in self.info.splits:
            # Open annotation files
            # annotation_paths = input_dirs["objects"]
            # Get images paths
            image_paths = []
            for ftype in ["*.png", "*.jpg", "*.jpeg"]:
                if split == "dataset":
                    image_paths.extend(glob.glob(str(input_dirs["image"] / ftype)))
                else:
                    image_paths.extend(
                        glob.glob(str(input_dirs["image"] / split / ftype))
                    )
            image_paths = [Path(p) for p in sorted(image_paths, key=natural_key)]
            
            annotation_path = input_dirs["objects"] / split / "dataframe.csv"
            annotations = pd.read_csv(annotation_path).to_numpy()[:,0].tolist()

            # import pdb
            # pdb.set_trace()

                        # Process rows
            for i in range(len(image_paths)):
                im_path= image_paths[i]

                # Create image thumbnail
                im_thumb = image_to_thumbnail(im_path.read_bytes())

                # Set image URI
                if portable:
                    im_uri = (
                        f"image/{im_path.name}"
                        if split == "dataset"
                        else f"image/{split}/{im_path.name}"
                    )
                else:
                    im_uri = im_path.absolute().as_uri()

                # Return rows
                rows = {
                    "main": {
                        "db": [
                            {
                                "id": im_path.name,
                                "views": ["image"],
                                "split": split,
                                "label": str(annotations[i]),
                            }
                        ]
                    },
                    "media": {
                        "image": [
                            {
                                "id": im_path.name,
                                "image": Image(im_uri, None, im_thumb).to_dict(),
                            }
                        ]
                    }#,
                    # "objects": {
                    #     "objects": [
                    #         {
                    #             "category_id": int(annotations[i]),
                    #             "category_name": str(annotations[i]),
                    #         }
                    #     ]
                    # },
                }
                yield rows