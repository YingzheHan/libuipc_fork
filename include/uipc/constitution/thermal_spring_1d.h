#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API ThermalSpring1D : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    ThermalSpring1D(const Json& config = default_config()) noexcept;

    // Public wrapper to mark this extra constitution on a geometry
    void apply_to(geometry::SimplicialComplex& sc) const;

    static Json default_config() noexcept;

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution
