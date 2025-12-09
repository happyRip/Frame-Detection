-- Simple JSON encoder for Lightroom plugins
-- Handles encoding Lua tables to JSON strings for the filter config

local JSON = {}

-- Escape special characters in strings for JSON
local function escapeString(s)
	s = s:gsub("\\", "\\\\")
	s = s:gsub('"', '\\"')
	s = s:gsub("\n", "\\n")
	s = s:gsub("\r", "\\r")
	s = s:gsub("\t", "\\t")
	return s
end

-- Check if table is an array (sequential integer keys starting at 1)
local function isArray(t)
	if type(t) ~= "table" then
		return false
	end
	local i = 0
	for _ in pairs(t) do
		i = i + 1
		if t[i] == nil then
			return false
		end
	end
	return true
end

-- Encode a Lua value to JSON string
local function encode(value, indent, currentIndent)
	indent = indent or 2
	currentIndent = currentIndent or 0

	local valueType = type(value)
	local nextIndent = currentIndent + indent
	local prefix = string.rep(" ", currentIndent)
	local innerPrefix = string.rep(" ", nextIndent)

	if value == nil then
		return "null"
	elseif valueType == "boolean" then
		return value and "true" or "false"
	elseif valueType == "number" then
		-- Handle special float cases
		if value ~= value then
			return "null" -- NaN
		elseif value == math.huge or value == -math.huge then
			return "null" -- Infinity
		end
		-- Format with appropriate precision
		if value == math.floor(value) then
			return string.format("%d", value)
		else
			return string.format("%.6g", value)
		end
	elseif valueType == "string" then
		return '"' .. escapeString(value) .. '"'
	elseif valueType == "table" then
		if isArray(value) then
			-- Encode as JSON array
			if #value == 0 then
				return "[]"
			end
			local items = {}
			for i, v in ipairs(value) do
				items[i] = innerPrefix .. encode(v, indent, nextIndent)
			end
			return "[\n" .. table.concat(items, ",\n") .. "\n" .. prefix .. "]"
		else
			-- Encode as JSON object
			local items = {}
			local keys = {}
			for k in pairs(value) do
				if type(k) == "string" then
					table.insert(keys, k)
				end
			end
			table.sort(keys) -- Sort keys for consistent output
			if #keys == 0 then
				return "{}"
			end
			for _, k in ipairs(keys) do
				local v = value[k]
				table.insert(items, innerPrefix .. '"' .. escapeString(k) .. '": ' .. encode(v, indent, nextIndent))
			end
			return "{\n" .. table.concat(items, ",\n") .. "\n" .. prefix .. "}"
		end
	else
		-- Unsupported type - return null with warning
		return "null"
	end
end

-- Public encode function with error handling
function JSON.encode(value)
	local success, result = pcall(encode, value, 2, 0)
	if success then
		return result, nil
	else
		return nil, "Failed to encode JSON: " .. tostring(result)
	end
end

-- Write JSON to file with error handling
function JSON.writeFile(path, value)
	local jsonStr, encodeErr = JSON.encode(value)
	if encodeErr then
		return false, encodeErr
	end

	local file, openErr = io.open(path, "w")
	if not file then
		return false, "Could not create JSON file: " .. tostring(openErr)
	end

	local writeSuccess, writeErr = file:write(jsonStr)
	file:close()

	if not writeSuccess then
		return false, "Failed to write JSON file: " .. tostring(writeErr)
	end

	return true, nil
end

return JSON
