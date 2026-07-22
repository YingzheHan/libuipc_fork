#include <contact_system/contact_reporter.h>

namespace uipc::backend::cuda
{
void ContactReporter::do_init(InitInfo&) {}

void ContactReporter::init()
{
    InitInfo info;
    do_init(info);
}

void ContactReporter::report_gradient_hessian_extent(GlobalContactManager::GradientHessianExtentInfo& info)
{
    do_report_gradient_hessian_extent(info);
    set_gradient_hessian_extent(info.gradient_count(), info.hessian_count());
}

void ContactReporter::assemble(GlobalContactManager::GradientHessianInfo& info)
{
    do_assemble(info);
    m_impl.gradients = info.gradients();
    m_impl.hessians  = info.hessians();
}

void ContactReporter::do_report_gradient_hessian_extent(GlobalDyTopoEffectManager::GradientHessianExtentInfo& info)
{
    GlobalContactManager::GradientHessianExtentInfo contact_info;
    contact_info.gradient_only(info.gradient_only());
    do_report_gradient_hessian_extent(contact_info);
    set_gradient_hessian_extent(contact_info.gradient_count(), contact_info.hessian_count());
    info.gradient_count(contact_info.gradient_count());
    info.hessian_count(contact_info.hessian_count());
}

void ContactReporter::do_assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info)
{
    GlobalContactManager::GradientHessianInfo contact_info;
    contact_info.gradient_only(info.gradient_only());
    contact_info.gradients(info.gradients());
    contact_info.hessians(info.hessians());
    do_assemble(contact_info);
    m_impl.gradients = contact_info.gradients();
    m_impl.hessians  = contact_info.hessians();
}

void ContactReporter::set_gradient_hessian_extent(SizeT gradient_count, SizeT hessian_count) noexcept
{
    m_impl.gradient_count = gradient_count;
    m_impl.hessian_count  = hessian_count;
}

SizeT ContactReporter::gradient_count() const noexcept
{
    return m_impl.gradient_count;
}

SizeT ContactReporter::hessian_count() const noexcept
{
    return m_impl.hessian_count;
}

void ContactReporter::do_build(DyTopoEffectReporter::BuildInfo& info)
{
    auto& manager = require<GlobalContactManager>();

    BuildInfo this_info;
    do_build(this_info);

    manager.add_reporter(this);
}
}  // namespace uipc::backend::cuda
