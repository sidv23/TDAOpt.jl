abstract type AbstractLossFunction end
abstract type AbstractDiscrepancy end
abstract type AbstractBackprop end
abstract type AbstractGradflow <: AbstractBackprop end

abstract type StatDiscrepancy <: AbstractDiscrepancy end
abstract type TDADiscrepancy <: AbstractDiscrepancy end