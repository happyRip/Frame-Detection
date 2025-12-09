-- Path management for AutoCrop plugin

local LrFileUtils = import("LrFileUtils")
local LrPathUtils = import("LrPathUtils")
local LrTasks = import("LrTasks")

-- System temp directory with plugin-specific subdirectory
local systemTemp = LrPathUtils.getStandardFilePath("temp")
local pluginTempRoot = LrPathUtils.child(systemTemp, _PLUGIN.id)

-- Temp directory paths
local renderPath = LrPathUtils.child(pluginTempRoot, "render")
local renderBackupPath = LrPathUtils.child(pluginTempRoot, "render.bak")
local debugPath = LrPathUtils.child(pluginTempRoot, "debug")
local previewPath = LrPathUtils.child(pluginTempRoot, "preview")
local previewBackupPath = LrPathUtils.child(pluginTempRoot, "preview.bak")

-- Symlink paths in plugin directory
local pluginRenderLink = LrPathUtils.child(_PLUGIN.path, "render")
local pluginDebugLink = LrPathUtils.child(_PLUGIN.path, "debug")
local pluginPreviewLink = LrPathUtils.child(_PLUGIN.path, "preview")

-- State for backup flow synchronization
local backupFlowComplete = true
local previewBackupFlowComplete = true

-- Generate a unique ID for temp directories
local function generateUid()
	return os.time() .. "-" .. math.random(10000, 99999)
end

-- Create a symlink (macOS/Windows)
local function createSymlink(target, link)
	-- Remove existing link/directory if it exists
	if LrFileUtils.exists(link) then
		-- Check if it's already a symlink pointing to the right place
		if MAC_ENV then
			local handle = io.popen('readlink "' .. link .. '"')
			if handle then
				local result = handle:read("*a")
				handle:close()
				if result and result:gsub("%s+", "") == target then
					return -- Already correct symlink
				end
			end
			LrTasks.execute('rm -rf "' .. link .. '"')
		else
			LrTasks.execute('rmdir /s /q "' .. link .. '"')
		end
	end
	-- Create symlink
	if MAC_ENV then
		LrTasks.execute('ln -s "' .. target .. '" "' .. link .. '"')
	else
		LrTasks.execute('mklink /D "' .. link .. '" "' .. target .. '"')
	end
end

-- Create directory if it doesn't exist
local function ensureDirectory(path)
	if not LrFileUtils.exists(path) then
		LrFileUtils.createDirectory(path)
	end
end

-- Delete a directory and its contents
local function deleteDirectory(path)
	if LrFileUtils.exists(path) then
		for filePath in LrFileUtils.files(path) do
			LrFileUtils.delete(filePath)
		end
		LrFileUtils.delete(path)
	end
end

-- Initialize temp directory structure and symlinks
local function init()
	ensureDirectory(pluginTempRoot)
	ensureDirectory(renderPath)
	ensureDirectory(debugPath)
	ensureDirectory(previewPath)
	createSymlink(renderPath, pluginRenderLink)
	createSymlink(debugPath, pluginDebugLink)
	createSymlink(previewPath, pluginPreviewLink)
end

-- Start backup flow (async): delete render.bak, rename render to render.bak
local function startBackupFlow()
	backupFlowComplete = false
	LrTasks.startAsyncTask(function()
		-- Delete old backup
		deleteDirectory(renderBackupPath)
		-- Rename current render to backup
		if LrFileUtils.exists(renderPath) then
			LrFileUtils.move(renderPath, renderBackupPath)
		end
		backupFlowComplete = true
	end)
end

-- Create a unique render temp directory for this run
local function createRenderTemp()
	ensureDirectory(pluginTempRoot)
	local tempPath = LrPathUtils.child(pluginTempRoot, "render_" .. generateUid())
	LrFileUtils.createDirectory(tempPath)
	return tempPath
end

-- Finalize render: wait for backup flow, then rename temp to render
local function finalizeRender(tempPath)
	-- Wait for backup flow to complete
	while not backupFlowComplete do
		LrTasks.sleep(0.1)
	end
	-- Rename temp to render
	if LrFileUtils.exists(tempPath) then
		LrFileUtils.move(tempPath, renderPath)
	end
	-- Recreate symlink
	createSymlink(renderPath, pluginRenderLink)
end

-- Start preview backup flow (async): delete preview.bak, rename preview to preview.bak
local function startPreviewBackupFlow()
	previewBackupFlowComplete = false
	LrTasks.startAsyncTask(function()
		-- Delete old backup
		deleteDirectory(previewBackupPath)
		-- Rename current preview to backup
		if LrFileUtils.exists(previewPath) then
			LrFileUtils.move(previewPath, previewBackupPath)
		end
		previewBackupFlowComplete = true
	end)
end

-- Create a unique preview temp directory for this run
local function createPreviewTemp()
	ensureDirectory(pluginTempRoot)
	local tempPath = LrPathUtils.child(pluginTempRoot, "preview_" .. generateUid())
	LrFileUtils.createDirectory(tempPath)
	return tempPath
end

-- Finalize preview: wait for backup flow, then rename temp to preview
local function finalizePreview(tempPath)
	-- Wait for backup flow to complete
	while not previewBackupFlowComplete do
		LrTasks.sleep(0.1)
	end
	-- Rename temp to preview
	if LrFileUtils.exists(tempPath) then
		LrFileUtils.move(tempPath, previewPath)
	end
	-- Recreate symlink
	createSymlink(previewPath, pluginPreviewLink)
end

return {
	-- Paths
	pluginTempRoot = pluginTempRoot,
	render = renderPath,
	renderBackup = renderBackupPath,
	debug = debugPath,
	preview = previewPath,
	previewBackup = previewBackupPath,
	pluginRenderLink = pluginRenderLink,
	pluginDebugLink = pluginDebugLink,
	pluginPreviewLink = pluginPreviewLink,

	-- Functions
	init = init,
	ensureDirectory = ensureDirectory,
	startBackupFlow = startBackupFlow,
	createRenderTemp = createRenderTemp,
	finalizeRender = finalizeRender,
	startPreviewBackupFlow = startPreviewBackupFlow,
	createPreviewTemp = createPreviewTemp,
	finalizePreview = finalizePreview,
}
