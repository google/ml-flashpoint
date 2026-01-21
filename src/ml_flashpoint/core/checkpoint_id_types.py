# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import re
from collections import UserString
from typing import Optional

from ml_flashpoint.core.defaults import DIRTY_MARKER_SUFFIX


class CheckpointId(UserString, abc.ABC):
    """
    A unique (fully-qualified) hierarchical identifier for a particular checkpoint resource in a directory path format.
    A resource can be a checkpoint container/namespace (i.e.a prefix or parent to data blobs), or an actual checkpoint
    data blob. Using a filesystem analogy, this would map to a directory or a file, respectively.

    Should be used via a subtype - see CheckpointContainerId and CheckpointObjectId.

    The ID must start with '/', and be hierarchical in that splitting by '/' and removing the last segment should
    allow for matching/finding all checkpoint containers in the same namespace (or session/job).
    All trailing '/' are removed (and thus are unnecessary/irrelevant): "/a/b/c/" is equivalent to "/a/b/c".

    Example: `/my/chkpt/18384/step=1-foo/_01_00.cp` may be the unique identifier for data blob
    `_01_00.cp` in the `step=1-foo` checkpoint container (namespace) within session ID 18384, and all other checkpoint
    versions in the same session will start with `/my/chkpt/18384/`, and within it all data blobs for `step=1-foo`
    will start with `/my/chkpt/18384/step=1--foo`.
    """

    @abc.abstractmethod
    def __init__(self, string: str, is_leaf_object: bool):
        """Initializes a CheckpointId.

        Args:
            string: The string representation of the ID.
            is_leaf_object: True if this ID represents a leaf object (e.g., a file), False otherwise.

        Raises:
            ValueError: If the string is invalid (e.g., does not start with '/', contains empty segments,
            or ends with '.').
        """
        if not string.startswith("/"):
            raise ValueError(f"A {self.__class__.__name__} must begin with '/', got '{string}' instead.")

        # At this point we know the string is 1) not empty and 2) begins with '/'

        # Normalize by removing trailing slashes, but handle the root case.
        stripped_string = string.rstrip("/") or "/"
        # Post-condition: String is either a. exactly '/', or b. is longer, starts with '/', and ends with a non-slash.

        if is_leaf_object and len(stripped_string) <= 1:
            raise ValueError(f"'{stripped_string}' is not a valid leaf ID.")

        if len(stripped_string) > 1:  # if stripped_string is not just '/', and there is more to validate
            parts = stripped_string.split("/")
            # parts[0] is empty string before the first '/'
            for part in parts[1:]:
                if not part:
                    raise ValueError(f"Path segment cannot be empty, got '{string}'")
                if part.endswith("."):
                    raise ValueError(f"Path segment cannot end with a '.', got '{string}'")

        super().__init__(stripped_string)

    def get_parent(self, levels_up: int = 1) -> "CheckpointContainerId | None":
        if levels_up < 1:
            raise ValueError(f"levels_up must be >= 1, got '{levels_up}' instead.")
        parts = self.data.split("/")
        if levels_up >= len(parts) - 1:  # if going to the root or higher
            return None
        return CheckpointContainerId("/".join(parts[:-levels_up]))


class CheckpointContainerId(CheckpointId):
    """
    Represents a checkpoint container or namespace, analogous to a directory.
    It acts as a container for checkpoint objects.
    This type is conceptually recursive - it may be a child or parent of other CheckpointContainerIds.
    """

    _VERSION_CONTAINER_PATTERN = re.compile(r"^step-(\d+)_ckpt$")
    _VERSION_CONTAINER_FORMAT_STRING = "step-{step}_ckpt"
    _DIRTY_VERSION_CONTAINER_PATTERN = re.compile(
        r"^step-(\d+)_ckpt.*_{}".format(re.escape(DIRTY_MARKER_SUFFIX)) + r"$"
    )

    def __init__(self, string: str):
        super().__init__(string, is_leaf_object=False)
        if self.data == "/":
            raise ValueError("CheckpointContainerId cannot be the root path '/'")

    @classmethod
    def get_version_container_pattern(cls) -> re.Pattern:
        """Returns the regex pattern for checkpoint directories.

        Returns:
            The regex pattern for checkpoint directories.
        """
        return cls._VERSION_CONTAINER_PATTERN

    @classmethod
    def get_dirty_version_checkpoint_container_pattern(cls) -> re.Pattern:
        """Returns the regex pattern for dirty checkpoint containers."""
        return cls._DIRTY_VERSION_CONTAINER_PATTERN

    @classmethod
    def format_version_container(cls, step: int) -> str:
        """Formats the checkpoint directory name for a given step.

        Args:
            step: The step number.

        Returns:
            The formatted checkpoint directory name.
        """
        return cls._VERSION_CONTAINER_FORMAT_STRING.format(step=step)

    @classmethod
    def parse_version_container_step(cls, name: str) -> Optional[int]:
        """
        Parses the step number from a checkpoint directory name.

        Args:
            name: The name of the directory.

        Returns:
            The step number if the name matches the pattern, otherwise None.
        """
        match = cls._VERSION_CONTAINER_PATTERN.match(name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def create_child(parent_container_id: "CheckpointContainerId", child_id: str) -> "CheckpointContainerId":
        """Creates a new CheckpointContainerId representing a child of the given parent container.

        Args:
            parent_container_id: The parent container ID.
            child_id: The ID string of the child container (e.g., a directory name).

        Returns:
            A new CheckpointContainerId representing the child container.
        """
        if not isinstance(parent_container_id, CheckpointContainerId):
            raise TypeError("parent_container_id must be a CheckpointContainerId")

        base = parent_container_id.data.rstrip("/")
        child = child_id.strip("/")

        if not child:
            # If child_id is empty or just slashes, return the parent
            return parent_container_id

        return CheckpointContainerId(f"{base}/{child}")


class CheckpointObjectId(CheckpointId):
    """
    Represents a specific checkpoint data blob, analogous to a file.
    This is a leaf node in the checkpoint hierarchy.
    """

    def __init__(self, string: str):
        super().__init__(string, is_leaf_object=True)

    @staticmethod
    def from_container(parent_container_id: CheckpointContainerId, object_name: str):
        if not isinstance(parent_container_id, CheckpointContainerId):
            raise TypeError("parent_container_id must be a CheckpointContainerId")

        base = parent_container_id.data.rstrip("/")
        child = object_name.strip().strip("/")

        if not child:
            raise ValueError("object_name must not be empty or consist only of whitespace and slashes")

        return CheckpointObjectId(f"{base}/{child}")
