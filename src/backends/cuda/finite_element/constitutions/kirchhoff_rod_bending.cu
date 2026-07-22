#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/kirchhoff_rod_bending_function.h>
#include <numbers>
#include <cmath>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <backends/cuda/utils/dump_utils.h>
#include <fmt/format.h>

#include <kernel_cout.h>
namespace uipc::backend::cuda
{
class KirchhoffRodBending final : public FiniteElementExtraConstitution
{
    static constexpr U64   KirchhoffRodBendingUID = 15;
    static constexpr SizeT StencilSize            = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override { return KirchhoffRodBendingUID; }

    vector<Vector3i> h_hinges;
    vector<Float>    h_bending_stiffness;

    muda::DeviceBuffer<Vector3i> hinges;
    muda::DeviceBuffer<Float>    bending_stiffnesses;
    muda::DeviceBuffer<Float>    hinge_curvatures;
    muda::DeviceBuffer<Float>    hinge_bend_strains;
    muda::DeviceBuffer<Float>    hinge_max_stresses;

    FiniteElementMethod* fem = nullptr;

    BufferDump dump_curvature;
    BufferDump dump_bend_strain;
    BufferDump dump_bend_stress;


    virtual void do_build(BuildInfo& info) override
    {
        fem = &require<FiniteElementMethod>();
    }

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();


        list<Vector3i> hinge_list;  // X0, X1, X2
        list<Float>    bending_stiffness_list;

        info.for_each(  //
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                unordered_map<IndexT, set<IndexT>> hinge_map;  // Vertex -> Connected Vertices

                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vertex_offset_v = vertex_offset->view().front();

                auto edges = sc.edges().topo().view();

                for(auto e : edges)
                {
                    auto v0 = e[0];
                    auto v1 = e[1];

                    hinge_map[v0].insert(v1);
                    hinge_map[v1].insert(v0);
                }

                auto bending_stiffnesses = sc.vertices().find<Float>("bending_stiffness");
                UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");

                auto bs_view = bending_stiffnesses->view();

                for(auto& [v, connected] : hinge_map)
                {
                    auto bs = bs_view[v];

                    if(connected.size() < 2)  // Not a hinge
                        continue;

                    for(auto v1 : connected)
                        for(auto v2 : connected)
                        {
                            if(v1 >= v2)  // Avoid duplicate
                                continue;

                            hinge_list.push_back({vertex_offset_v + v1,
                                                  vertex_offset_v + v,  // center vertex
                                                  vertex_offset_v + v2});
                            bending_stiffness_list.push_back(bs);
                        }
                }
            });

        // Setup data
        h_hinges.resize(hinge_list.size());
        h_bending_stiffness.resize(hinge_list.size());
        std::ranges::move(hinge_list, h_hinges.begin());
        std::ranges::move(bending_stiffness_list, h_bending_stiffness.begin());

        // Copy to device
        hinges.resize(h_hinges.size());
        hinges.view().copy_from(h_hinges.data());

        bending_stiffnesses.resize(h_bending_stiffness.size());
        bending_stiffnesses.view().copy_from(h_bending_stiffness.data());

        hinge_curvatures.resize(hinges.size(), 0.0);
        hinge_bend_strains.resize(hinges.size(), 0.0);
        hinge_max_stresses.resize(hinges.size(), 0.0);
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(hinges.size());  // Each hinge has 1 energy
        info.gradient_count(hinges.size() * StencilSize);  // Each hinge has 3 vertices

        if(info.gradient_only())
            return;

        info.hessian_count(hinges.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace KRB = sym::kirchhoff_rod_bending;

        constexpr Float Pi = std::numbers::pi;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.energies().size(),
                   [hinges = hinges.viewer().name("hinges"),
                    bending_stiffnesses = bending_stiffnesses.viewer().name("bending_stiffness"),
                    thicknesses = info.thicknesses().viewer().name("thickness"),
                    xs          = info.xs().viewer().name("xs"),
                    x_bars      = info.x_bars().viewer().name("x_bars"),
                    energies    = info.energies().viewer().name("energies"),
                    dt          = info.dt(),
                    Pi] __device__(int I)
                   {
                       Vector3i hinge = hinges(I);
                       Float    k     = bending_stiffnesses(I) * dt * dt;
                       // thicknesses is indexed by global vertex id, not hinge id.
                       Float    r     = thicknesses(hinge[1]);

                       Vector9 X;
                       X.segment<3>(0) = xs(hinge[0]);
                       X.segment<3>(3) = xs(hinge[1]);
                       X.segment<3>(6) = xs(hinge[2]);

                       Vector3 x0_bar = x_bars(hinge[0]);
                       Vector3 x1_bar = x_bars(hinge[1]);
                       Vector3 x2_bar = x_bars(hinge[2]);

                       // Rest length of the two edges
                       Float L0 = (x1_bar - x0_bar).norm() + (x2_bar - x1_bar).norm();

                       Float E;
                       KRB::E(E, k, X, L0, r, Pi);

                       energies(I) = E;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace KRB = sym::kirchhoff_rod_bending;

        constexpr Float Pi = std::numbers::pi;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(hinges.size(),
                   [hinges = hinges.viewer().name("hinges"),
                    bending_stiffnesses = bending_stiffnesses.viewer().name("bending_stiffness"),
                    thicknesses = info.thicknesses().viewer().name("thickness"),
                    xs          = info.xs().viewer().name("xs"),
                    x_bars      = info.x_bars().viewer().name("x_bars"),
                    G3s         = info.gradients().viewer().name("gradients"),
                    H3x3s       = info.hessians().viewer().name("hessians"),
                    dt          = info.dt(),
                    Pi,
                    gradient_only = info.gradient_only()] __device__(int I) mutable
                   {
                       Vector3i hinge = hinges(I);
                       Float    k     = bending_stiffnesses(I);
                       // thicknesses is indexed by global vertex id, not hinge id.
                       Float    r     = thicknesses(hinge[1]);

                       Vector9 X;
                       X.segment<3>(0) = xs(hinge[0]);
                       X.segment<3>(3) = xs(hinge[1]);
                       X.segment<3>(6) = xs(hinge[2]);

                       Vector3 x0_bar = x_bars(hinge[0]);
                       Vector3 x1_bar = x_bars(hinge[1]);
                       Vector3 x2_bar = x_bars(hinge[2]);

                       // Rest length of the two edges
                       Float L0 = (x1_bar - x0_bar).norm() + (x2_bar - x1_bar).norm();

                       Float dt2 = dt * dt;

                       Vector9 G;
                       KRB::dEdX(G, k, X, L0, r, Pi);
                       G *= dt2;
                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<StencilSize>(I * StencilSize).write(hinge, G);

                       if(gradient_only)
                           return;

                       Matrix9x9 H;
                       KRB::ddEddX(H, k, X, L0, r, Pi);

                       H *= dt2;
                       make_spd(H);
                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.half_block<StencilSize>(I * HalfHessianSize).write(hinge, H);
                   });
    }

    virtual bool do_dump(DumpInfo& info) override
    {
        if(!fem)
            return true;

        auto path  = info.dump_path(__FILE__);
        auto frame = info.frame();

        if(hinges.size() > 0)
        {
            using namespace muda;

            auto xs_view        = fem->xs();
            auto x_bars_view    = fem->x_bars();
            auto thickness_view = fem->thicknesses();

            auto xs          = xs_view.viewer().name("xs");
            auto x_bars      = x_bars_view.viewer().name("x_bars");
            auto thicknesses = thickness_view.viewer().name("thicknesses");

            auto hinge_view       = hinges.cviewer().name("hinges");
            auto bending_view     = bending_stiffnesses.cviewer().name("bending_stiffnesses");
            auto curvature_view   = hinge_curvatures.viewer().name("hinge_curvatures");
            auto bend_strain_view = hinge_bend_strains.viewer().name("hinge_bend_strains");
            auto max_stress_view  = hinge_max_stresses.viewer().name("hinge_max_stresses");

            ParallelFor()
                .kernel_name("KirchhoffRodBending Dump")
                .apply(hinges.size(),
                       [hinges       = hinge_view,
                        xs           = xs,
                        x_bars       = x_bars,
                        thicknesses  = thicknesses,
                        bending      = bending_view,
                        curvatures   = curvature_view,
                        bend_strains = bend_strain_view,
                        max_stresses = max_stress_view] __device__(int I) mutable
                       {
                           const Vector3i hinge = hinges(I);
                           const IndexT   v0    = hinge[0];
                           const IndexT   vc    = hinge[1];
                           const IndexT   v2    = hinge[2];

                           const Vector3& x0 = xs(v0);
                           const Vector3& xc = xs(vc);
                           const Vector3& x2 = xs(v2);

                           const Vector3& X0 = x_bars(v0);
                           const Vector3& Xc = x_bars(vc);
                           const Vector3& X2 = x_bars(v2);

                           const Float thickness = thicknesses(vc);
                           const Float stiffness = bending(I);

                           const Vector3 e0     = x0 - xc;
                           const Vector3 e1     = x2 - xc;
                           const Float   e0_len = e0.norm();
                           const Float   e1_len = e1.norm();

                           Float phi = 0.0;
                           if(e0_len > static_cast<Float>(1e-12) && e1_len > static_cast<Float>(1e-12))
                           {
                               Float cos_phi = (e0.dot(e1)) / (e0_len * e1_len);
                               if(cos_phi > static_cast<Float>(1.0))
                                   cos_phi = static_cast<Float>(1.0);
                               else if(cos_phi < static_cast<Float>(-1.0))
                                   cos_phi = static_cast<Float>(-1.0);
                               phi = acos(cos_phi);
                           }

                           const Float L0 = (Xc - X0).norm() + (X2 - Xc).norm();

                           Float curvature = 0.0;
                           if(L0 > static_cast<Float>(1e-12))
                               curvature = static_cast<Float>(2.0) * sin(phi * static_cast<Float>(0.5)) / L0;

                           const Float bend_strain = curvature * thickness;
                           const Float max_stress  = stiffness * bend_strain;

                           curvatures(I)   = curvature;
                           bend_strains(I) = bend_strain;
                           max_stresses(I) = max_stress;
                       });
        }

        bool ok = true;

        if(hinges.size() > 0)
        {
            ok = ok
                 && dump_curvature.dump(fmt::format("{}rod_curvature.{}", path, frame), hinge_curvatures);
            ok = ok
                 && dump_bend_strain.dump(fmt::format("{}rod_bend_strain.{}", path, frame),
                                          hinge_bend_strains);
            ok = ok
                 && dump_bend_stress.dump(fmt::format("{}rod_bend_stress.{}", path, frame),
                                          hinge_max_stresses);
        }

        return ok;
    }
};


REGISTER_SIM_SYSTEM(KirchhoffRodBending);
}  // namespace uipc::backend::cuda
