// Thermal axial expansion/shrinkage for 1D rods as an extra constitution.
// Adds an energy term like Hookean spring but with a thermal target length
//   L0_eff = (1 + eps_th) * L0,  eps_th = alpha * (T - T_ref) (averaged per edge).
// This keeps material/dynamics intact and explicitly couples thermal strain in
// the nonlinear solve each iteration.

#include <finite_element/finite_element_extra_constitution.h>
#include <kernel_cout.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <utils/codim_thickness.h>

#include <finite_element/constitutions/hookean_spring_1d_function.h>
#include <uipc/builtin/attribute_name.h>
#include <list>
#include <ranges>

namespace uipc::backend::cuda
{
// Choose a new UID not conflicting with existing ones (12,14,15 used).
static constexpr U64 ThermalSpring1DUID = 60ull;
static constexpr SizeT StencilSize      = 2;
static constexpr SizeT HessianSize      = StencilSize * StencilSize;

class ThermalSpring1D final : public FiniteElementExtraConstitution
{
  public:
    using Base = FiniteElementExtraConstitution;
    using Base::Base;

    // host buffers
    vector<Vector2i> h_edges;     // global vertex indices per edge
    vector<Float>    h_kappas;    // per-edge axial stiffness kappa (EA)
    vector<Float>    h_eps_th;    // per-edge thermal strain eps = alpha*(T - T_ref)

    // device buffers
    muda::DeviceBuffer<Vector2i> edges;
    muda::DeviceBuffer<Float>    kappas;
    muda::DeviceBuffer<Float>    eps_ths;

    virtual U64 get_uid() const noexcept override { return ThermalSpring1DUID; }

    virtual void do_build(BuildInfo&) override
    {
        // Update thermal strain from scene attributes at the beginning of each frame
        on_rebuild_scene([this]
                        {
                            // auto world_vis = world();
                            auto& world_vis = world();

                            auto geo_slots = world_vis.scene().geometries();

                            // Refresh per-edge thermal strain from per-vertex attributes
                            SizeT cursor = 0;
                            h_eps_th.resize(h_edges.size());

                            for(auto& geo_info : geo_infos())
                            {
                                auto& geo_slot = geo_slots[geo_info.geo_slot_index];
                                auto& geo      = geo_slot->geometry();
                                auto* sc       = geo.as<geometry::SimplicialComplex>();
                                if(!sc)
                                    continue;

                                auto vertex_offset = sc->meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                                UIPC_ASSERT(vertex_offset, "backend_fem_vertex_offset missing");
                                const IndexT v_off = vertex_offset->view().front();

                                auto edges_view = sc->edges().topo().view();

                                // per-vertex attributes (optional)
                                auto a_attr   = sc->vertices().find<Float>("thermal_alpha");
                                auto t_attr   = sc->vertices().find<Float>("temperature");
                                auto tr_attr  = sc->vertices().find<Float>("temperature_ref");

                                auto a_view  = a_attr ? a_attr->view() : span<const Float>();
                                auto t_view  = t_attr ? t_attr->view() : span<const Float>();
                                auto tr_view = tr_attr ? tr_attr->view() : span<const Float>();

                                for(SizeT i = 0; i < edges_view.size(); ++i)
                                {
                                    const auto e_local = edges_view[i];
                                    const IndexT v0 = v_off + e_local[0];
                                    const IndexT v1 = v_off + e_local[1];

                                    Float eps0 = 0.0f, eps1 = 0.0f;
                                    if(a_attr && t_attr && tr_attr)
                                    {
                                        eps0 = a_view[e_local[0]] * (t_view[e_local[0]] - tr_view[e_local[0]]);
                                        eps1 = a_view[e_local[1]] * (t_view[e_local[1]] - tr_view[e_local[1]]);
                                    }
                                    // simple average per edge
                                    h_eps_th[cursor++] = 0.5f * (eps0 + eps1);
                                }
                            }

                            if(!h_eps_th.empty())
                            {
                                eps_ths.resize(h_eps_th.size());
                                eps_ths.view().copy_from(h_eps_th.data());
                            }
                        });
    }

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        // auto world_vis = world();
        auto& world_vis = world();
        auto geo_slots = world_vis.scene().geometries();

        // Build per-edge lists and material params
        list<Vector2i> edge_list;
        list<Float>    kappa_list;
        list<Float>    eps_list;

        info.for_each(  // iterate over edges per geometry
            geo_slots,
            [](geometry::SimplicialComplex& sc) { return sc.edges().topo().view(); },
            [&](const ForEachInfo& I, const Vector2i& e_local)
            {
                auto& geo_info = I.geo_info();
                auto& geo_slot = geo_slots[geo_info.geo_slot_index];
                auto& geo      = geo_slot->geometry();
                auto* sc       = geo.as<geometry::SimplicialComplex>();

                auto vertex_offset = sc->meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "backend_fem_vertex_offset missing");
                const IndexT v_off = vertex_offset->view().front();

                // global vertex ids for this edge
                Vector2i edge_global{v_off + e_local[0], v_off + e_local[1]};
                edge_list.push_back(edge_global);

                // axial stiffness per edge (reuse HookeanSpring's "kappa" if present)
                Float kappa = 0.0f;
                if(auto k_attr = sc->edges().find<Float>("kappa"))
                {
                    auto kv = k_attr->view();
                    kappa   = kv[I.local_index()];
                }
                kappa_list.push_back(kappa);

                // initial eps_th from vertex attributes if already set (else 0)
                Float eps = 0.0f;
                if(auto a_attr = sc->vertices().find<Float>("thermal_alpha"))
                {
                    auto t_attr  = sc->vertices().find<Float>("temperature");
                    auto tr_attr = sc->vertices().find<Float>("temperature_ref");
                    if(t_attr && tr_attr)
                    {
                        auto av = a_attr->view();
                        auto tv = t_attr->view();
                        auto rv = tr_attr->view();
                        Float eps0 = av[e_local[0]] * (tv[e_local[0]] - rv[e_local[0]]);
                        Float eps1 = av[e_local[1]] * (tv[e_local[1]] - rv[e_local[1]]);
                        eps        = 0.5f * (eps0 + eps1);
                    }
                }
                eps_list.push_back(eps);
            });

        // Copy to device
        h_edges.resize(edge_list.size());
        h_kappas.resize(edge_list.size());
        h_eps_th.resize(edge_list.size());
        std::ranges::move(edge_list, h_edges.begin());
        std::ranges::move(kappa_list, h_kappas.begin());
        std::ranges::move(eps_list, h_eps_th.begin());

        edges.resize(h_edges.size());
        edges.view().copy_from(h_edges.data());

        kappas.resize(h_kappas.size());
        kappas.view().copy_from(h_kappas.data());

        eps_ths.resize(h_eps_th.size());
        eps_ths.view().copy_from(h_eps_th.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(edges.size());
        info.gradient_count(edges.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(edges.size() * HessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace NS = sym::hookean_spring_1d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(edges.size(),
                   [indices    = edges.cviewer().name("thermal_edges"),
                    kappas     = kappas.cviewer().name("kappas"),
                    eps_ths    = eps_ths.cviewer().name("eps_ths"),
                    xs         = info.xs().viewer().name("xs"),
                    x_bars     = info.x_bars().viewer().name("x_bars"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    energies   = info.energies().viewer().name("energies"),
                    dt         = info.dt(),
                    Pi         = std::numbers::pi] __device__(int I)
                   {
                       const Vector2i idx = indices(I);
                       Vector6 X;
                       X.segment<3>(0) = xs(idx(0));
                       X.segment<3>(3) = xs(idx(1));

                       // rest length from x_bars
                       Float L0 = (x_bars(idx(1)) - x_bars(idx(0))).norm();
                       const Float eps = eps_ths(I);
                       const Float L0_eff = (Float{1.0} + eps) * L0;

                       const Float r = edge_thickness(thicknesses(idx(0)), thicknesses(idx(1)));
                       const Float k = kappas(I);
                       const Float Vdt2 = L0 * r * r * Pi * dt * dt;

                       Float E;
                       NS::E(E, k, X, L0_eff);
                       energies(I) = E * Vdt2;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace NS = sym::hookean_spring_1d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(edges.size(),
                   [indices    = edges.cviewer().name("thermal_edges"),
                    kappas     = kappas.cviewer().name("kappas"),
                    eps_ths    = eps_ths.cviewer().name("eps_ths"),
                    xs         = info.xs().viewer().name("xs"),
                    x_bars     = info.x_bars().viewer().name("x_bars"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                    G3s        = info.gradients().viewer().name("gradients"),
                    H3x3s      = info.hessians().viewer().name("hessians"),
                    dt         = info.dt(),
                    Pi         = std::numbers::pi,
                    gradient_only = info.gradient_only()] __device__(int I) mutable
                   {
                       const Vector2i idx = indices(I);
                       Vector6 X;
                       X.segment<3>(0) = xs(idx(0));
                       X.segment<3>(3) = xs(idx(1));

                       // rest length from x_bars
                       Float L0 = (x_bars(idx(1)) - x_bars(idx(0))).norm();
                       const Float eps = eps_ths(I);
                       const Float L0_eff = (Float{1.0} + eps) * L0;

                       const Float r = edge_thickness(thicknesses(idx(0)), thicknesses(idx(1)));
                       const Float k = kappas(I);
                       const Float Vdt2 = L0 * r * r * Pi * dt * dt;

                       const Vector2i ignore = {is_fixed(idx(0)), is_fixed(idx(1))};

                       Vector6 G;
                       NS::dEdX(G, k, X, L0_eff);
                       G *= Vdt2;
                       DoubletVectorAssembler VA{G3s};
                       VA.segment<2>(I * 2).write(idx, ignore, G);

                       if(!gradient_only)
                       {
                           Matrix6x6 H;
                           NS::ddEddX(H, k, X, L0_eff);
                           H *= Vdt2;
                           make_spd(H);
                           TripletMatrixAssembler MA{H3x3s};
                           MA.block<StencilSize, StencilSize>(I * HessianSize).write(idx, ignore, H);
                       }
                   });
    }
};

REGISTER_SIM_SYSTEM(ThermalSpring1D);
}  // namespace uipc::backend::cuda
