#include <uipc/common/config.h>
#include <uipc/common/log.h>
#include <backends/common/backend_path_tool.h>

namespace uipc::backend
{
BackendPathTool::BackendPathTool(std::string_view workspace) noexcept
    : m_workspace(workspace)
{
}
auto BackendPathTool::workspace() const noexcept -> Path
{
    return m_workspace;
}

auto BackendPathTool::workspace(std::string_view uipc_source_file,
                                std::string_view prefix) const noexcept -> Path
{
    namespace fs = std::filesystem;

    Path file_path{uipc_source_file};
    Path backend_source_dir{UIPC_BACKEND_DIR};
    Path abs_file_path = file_path.is_absolute() ? file_path : Path{UIPC_PROJECT_DIR} / file_path;
    Path relative_to_backend_source_dir =
        fs::relative(abs_file_path, backend_source_dir);

    auto first_component = relative_to_backend_source_dir.begin();
    UIPC_ASSERT(first_component == relative_to_backend_source_dir.end()
                    || first_component->string() != "..",
                "Backend source file must be under {}, got {}",
                backend_source_dir.string(),
                abs_file_path.string());

    Path file_output_path = Path{m_workspace} / Path{prefix} / relative_to_backend_source_dir;

    // create all the intermediate directories if they don't exist
    if(!fs::exists(file_output_path))
        fs::create_directories(file_output_path);

    return (file_output_path / "");
}
}  // namespace uipc::backend
