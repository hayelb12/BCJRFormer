from enum import Enum, StrEnum
from typing import List
from spock import spock
from bcjrformer.codes.linear_code import CodeFileExtension, LinearCodeType
from bcjrformer.configs.specific_model_config import (
    BCJRFormerModel,
    CombinedConvBCJRFormerModel,
    ConvBCJRFormerModel,
    EcctOnBCJRFormerE2EModel,
    IdsECCTOuterModel,
    SpecificModelConfig,
)


class InnerCodeType(StrEnum):
    CONVOLUTIONAL = "convolutional"
    MARKER = "marker"


@spock
class ConvolutionalCodeConfig:
    """
    Convolutional Code Configuration

    Attributes:
        k: The number of information bits
        g: The generator polynomials
        p: The number of bits to encode the alphabet
        random_offset: Whether to randomly offset codeword
    """

    # Defaults let Spock instantiate this class when a marker (not convolutional)
    # inner code is selected; the selected inner code always overrides these.
    k: int = 1
    g: List[int] = [5, 7]
    p: int = 1
    random_offset: bool = True


@spock
class MarkerCodeConfig:
    """Marker Code Configuration

    Attributes:
        marker: The marker sequence
        N_c: Number of bits between two markers
        p: Number of bits to encode the alphabet
    """

    # Defaults let Spock instantiate this class when a convolutional (not marker)
    # inner code is selected; the selected inner code always overrides these.

    # the marker sequence
    marker: List[int] = [1, 1, 0]

    # number of bits between two markers
    N_c: int = 3

    # number of bits to encode the alphabet
    p: int = 1


class InnerCodeConfig(Enum):
    CONVOLUTIONAL = ConvolutionalCodeConfig
    MARKER = MarkerCodeConfig


@spock
class LinearCodeConfig:
    """Linear Code Configuration

    Attributes:
        code_type: Type of code
        k: Linear code k
        n: Linear code n
        q: Alphabet/Field size
        custom_file_name: Custom file name of the code's pc matrix
        file_extension: File extension of the code's pc matrix
    """

    code_type: LinearCodeType = LinearCodeType.LDPC
    k: int
    n: int
    q: int = 2
    custom_file_name: str | None = None
    random_weights: bool = False
    file_extension: CodeFileExtension | None = None


@spock
class ConcatenatedCodeConfig:
    """Either an outer code or a combination of inner and outer codes"""

    outer_code_config: LinearCodeConfig
    inner_code_config: InnerCodeConfig | None = None


InnerCodeConfigType = ConvolutionalCodeConfig | MarkerCodeConfig


def code_config_to_code_type(inner_code_config: InnerCodeConfigType) -> InnerCodeType:
    if isinstance(inner_code_config, ConvolutionalCodeConfig):
        return InnerCodeType.CONVOLUTIONAL
    elif isinstance(inner_code_config, MarkerCodeConfig):
        return InnerCodeType.MARKER
    else:
        raise ValueError(f"Invalid inner code config: {inner_code_config}")


# Allowed inner code type(s) per specific-model class. Pairing a model with a
# code config whose declared inner scheme is not in this set means Spock would
# otherwise silently fill the unselected inner class with defaults and train on
# garbage, so this is enforced as a hard fail-fast check.
_MODEL_ALLOWED_INNER_CODE_TYPES: dict[type, set[InnerCodeType]] = {
    BCJRFormerModel: {InnerCodeType.MARKER, InnerCodeType.CONVOLUTIONAL},
    ConvBCJRFormerModel: {InnerCodeType.CONVOLUTIONAL},
    CombinedConvBCJRFormerModel: {InnerCodeType.CONVOLUTIONAL},
    EcctOnBCJRFormerE2EModel: {InnerCodeType.MARKER},
    IdsECCTOuterModel: {InnerCodeType.MARKER, InnerCodeType.CONVOLUTIONAL},
}


def validate_model_code_compatibility(
    specific_model_config: SpecificModelConfig,
    inner_code_config: InnerCodeConfigType | None,
) -> None:
    """Fail fast if the selected model and the declared inner code scheme are
    incompatible.

    Spock silently instantiates the unselected inner code class with defaults,
    so without this check a model that needs a Convolutional inner code paired
    with a Marker-declared code config (or vice versa) would train on garbage
    instead of erroring.
    """
    if inner_code_config is None:
        raise ValueError(
            "Code config does not declare `inner_code_config` - pair a code "
            "config from configs/code/ that declares its inner scheme"
        )

    code_type = code_config_to_code_type(inner_code_config)

    allowed = _MODEL_ALLOWED_INNER_CODE_TYPES.get(type(specific_model_config))
    if allowed is None:
        raise ValueError(
            f"Unknown specific model {type(specific_model_config).__name__}: "
            "cannot validate model/code compatibility"
        )

    if code_type not in allowed:
        raise ValueError(
            f"Model {type(specific_model_config).__name__} requires inner code "
            f"type(s) {sorted(allowed)}, but the code config declares "
            f"{code_type}. Pair a compatible configs/code/ file."
        )
