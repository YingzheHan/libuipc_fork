#include <pyuipc/constitution/thermal_spring_1d.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/thermal_spring_1d.h>
#include <pyuipc/common/json.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;
PyThermalSpring1D::PyThermalSpring1D(py::module& m)
{
    auto class_ThermalSpring1D =
        py::class_<ThermalSpring1D, FiniteElementExtraConstitution>(m, "ThermalSpring1D");

    class_ThermalSpring1D.def(py::init<const Json&>(),
                              py::arg("config") = ThermalSpring1D::default_config());

    class_ThermalSpring1D.def_static("default_config", &ThermalSpring1D::default_config);

    // expose public apply_to wrapper
    class_ThermalSpring1D.def("apply_to", &ThermalSpring1D::apply_to, py::arg("sc"));
}
}  // namespace pyuipc::constitution
