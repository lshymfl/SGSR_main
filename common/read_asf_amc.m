function read_asf_amc(asf_file, amc_file, save_name)
%READ_ASF_AMC 读取 Acclaim 动作捕捉文件 (.asf & .amc)
%
%   read_asf_amc('subject.asf', 'subject.amc', 'output.mat')
%
%   输出内容将保存在 output.mat 中，包括骨架结构和动作数据。
%
%   作者：ChatGPT
%   日期：2025-11-10

    %% 检查输入
    if nargin < 3
        error('用法: read_asf_amc(asf_file, amc_file, save_name)');
    end

    %% 读取 ASF 文件（骨架结构）
    fprintf('正在读取骨架文件: %s\n', asf_file);
    asf_data = read_asf(asf_file);

    %% 读取 AMC 文件（动作数据）
    fprintf('正在读取动作文件: %s\n', amc_file);
    amc_data = read_amc(amc_file);

    %% 保存结果
    save(save_name, 'asf_data', 'amc_data');
    fprintf('数据已保存至 %s\n', save_name);
end

%% ============ ASF 文件读取函数 ============
function asf = read_asf(filename)
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开 ASF 文件: %s', filename);
    end
    
    asf.version = '';
    asf.name = '';
    asf.units = struct();
    asf.bones = struct([]);
    
    current_section = '';
    bone_index = 0;
    
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if startsWith(line, ':')
            current_section = lower(strrep(line(2:end), ' ', ''));
            continue;
        end

        if isempty(line) || startsWith(line, '#')
            continue; % 跳过空行和注释
        end

        switch current_section
            case 'version'
                asf.version = line;
            case 'name'
                asf.name = line;
            case 'units'
                parts = strsplit(line);
                if numel(parts) == 2
                    asf.units.(parts{1}) = str2double(parts{2});
                end
            case 'bonedata'
                if startsWith(line, 'begin')
                    bone_index = bone_index + 1;
                    bone = struct();
                elseif startsWith(line, 'end')
                    asf.bones(bone_index) = bone;
                else
                    parts = strsplit(line);
                    if numel(parts) >= 2
                        key = lower(parts{1});
                        value = strjoin(parts(2:end), ' ');
                        bone.(key) = value;
                    end
                end
            otherwise
                % 其他段落可以根据需要扩展
        end
    end
    
    fclose(fid);
end

%% ============ AMC 文件读取函数 ============
function amc = read_amc(filename)
    fid = fopen(filename, 'r');
    if fid == -1
        error('无法打开 AMC 文件: %s', filename);
    end

    amc.frames = {};
    frame_idx = 0;
    current_frame = struct();

    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if isempty(line) || startsWith(line, '#')
            continue;
        end
        
        % 检测是否为帧号
        num = str2double(line);
        if ~isnan(num)
            % 保存上一个帧
            if ~isempty(fieldnames(current_frame))
                amc.frames{frame_idx} = current_frame;
            end
            frame_idx = num;
            current_frame = struct();
        else
            % 动作数据行
            parts = strsplit(line);
            bone_name = parts{1};
            data = str2double(parts(2:end));
            current_frame.(bone_name) = data;
        end
    end
    
    % 保存最后一帧
    if ~isempty(fieldnames(current_frame))
        amc.frames{frame_idx} = current_frame;
    end
    
    amc.nFrames = length(amc.frames);
    fclose(fid);
end
