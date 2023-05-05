# Copyright 2020 Adap GmbH. All Rights Reserved.
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
# ==============================================================================
"""Flower NumPyClient tests."""


from typing import Dict, Tuple

from flwr.common import Config, NDArrays, Properties, Scalar

from .numpy_client import (
    NumPyClient,
    has_evaluate,
    has_fit,
    has_get_parameters,
    has_get_properties,
)


class OverridingClient(NumPyClient):
    """Client overriding `get_properties`."""

    def get_properties(self, config: Config) -> Properties:
        return Properties()

    def get_parameters(self, config: Config) -> NDArrays:
        return []

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return [], 0, {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return 0.0, 0, {}


class NotOverridingClient(NumPyClient):
    """Client not overriding any NumPyClient method."""


def test_has_get_properties_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_properties_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_parameters_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_parameters(client=client)

    # Assert
    assert actual == expected


def test_has_get_parameters_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_parameters(client=client)

    # Assert
    assert actual == expected


def test_has_fit_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_fit(client=client)

    # Assert
    assert actual == expected


def test_has_fit_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_fit(client=client)

    # Assert
    assert actual == expected


def test_has_evaluate_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_evaluate(client=client)

    # Assert
    assert actual == expected


def test_has_evaluate_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_evaluate(client=client)

    # Assert
    assert actual == expected
