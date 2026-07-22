#pragma once
#include <sim_system.h>
#include <contact_system/global_contact_manager.h>
#include <dytopo_effect_system/dytopo_effect_reporter.h>

namespace uipc::backend::cuda
{
class ContactReporter : public DyTopoEffectReporter
{
  public:
    using DyTopoEffectReporter::DyTopoEffectReporter;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    class Impl
    {
      public:
        muda::CBufferView<Float>           energies;
        muda::CDoubletVectorView<Float, 3> gradients;
        muda::CTripletMatrixView<Float, 3> hessians;
        SizeT                              gradient_count = 0;
        SizeT                              hessian_count  = 0;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info);
    void set_gradient_hessian_extent(SizeT gradient_count, SizeT hessian_count) noexcept;
    virtual void do_report_gradient_hessian_extent(GlobalContactManager::GradientHessianExtentInfo& info) = 0;
    virtual void do_assemble(GlobalContactManager::GradientHessianInfo& info)                     = 0;

  private:
    friend class GlobalContactManager;
    void  init();  // only be called by GlobalContactManager
    void  report_gradient_hessian_extent(GlobalContactManager::GradientHessianExtentInfo& info);
    void  assemble(GlobalContactManager::GradientHessianInfo& info);
    SizeT gradient_count() const noexcept;
    SizeT hessian_count() const noexcept;
    void  do_build(DyTopoEffectReporter::BuildInfo&) final override;
    virtual EnergyComponentFlags component_flags() override final
    {
        return EnergyComponentFlags::Contact;
    }
    void  do_report_gradient_hessian_extent(GlobalDyTopoEffectManager::GradientHessianExtentInfo& info) final override;
    void  do_assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info) final override;
    SizeT m_index = ~0ull;
    Impl  m_impl;
};
}  // namespace uipc::backend::cuda
