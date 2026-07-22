#include <uipc/constitution/thermal_spring_1d.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_auto_register.h>

namespace uipc::constitution
{
// Keep in sync with backend UID in cuda ThermalSpring1D
constexpr U64 ThermalSpring1DUID = 60ull;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = ThermalSpring1DUID;
    info.name = "ThermalSpring1D";
    info.type = string{builtin::FiniteElement};
    uid_infos.push_back(info);
    return uid_infos;
}

ThermalSpring1D::ThermalSpring1D(const Json& config) noexcept
    : m_config(config)
{
}

void ThermalSpring1D::apply_to(geometry::SimplicialComplex& sc) const
{
    // reuse Base behavior to append UID to geometry meta
    Base::apply_to(sc);
}

Json ThermalSpring1D::default_config() noexcept
{
    return Json::object();
}

U64 ThermalSpring1D::get_uid() const noexcept
{
    return ThermalSpring1DUID;
}
}  // namespace uipc::constitution
