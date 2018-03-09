function all_tasks = MakeNimsLookup(nims_dir,study,skip_renaming,init_name)
    if nargin < 4
        init_name = 'TASKNAME';
    else
    end
    if nargin < 3
        skip_renaming = false;
    else
    end
    if nargin < 2
        mat_file = sprintf('/Volumes/svndl/RAW_DATA/correction_files/untitled_correction.mat');
    else
        mat_file = sprintf('/Volumes/svndl/RAW_DATA/correction_files/%s_correction.mat',study);
    end
    if exist(mat_file,'file')
        json_data = load(mat_file);
    else
        if skip_renaming
            msg = sprintf('Cannot skip renaming because json_data mat-file "%s" does not exist\n',mat_file);
            error(msg);
        else
        end
        json_data = struct([]);
    end
    if ~skip_renaming
        sub_dirs = subfolders(nims_dir,1);
        sub_dirs = flip(sub_dirs); % start from latest
        for s = 1:length(sub_dirs)
            session_id = split_string(sub_dirs{s},'_');
            session_id = session_id{end};
            if isfield(json_data, ['session_',session_id])
                sub_id = json_data.(['session_',session_id]).sub_id;
            else
                sub_id = '0000';
            end
            if ~iscell(sub_id)
                sub_id = {sub_id};
            else
            end
            sub_id = inputdlg(session_id,'sub ID?',1,sub_id);
            if isempty(sub_id) || strcmp(sub_id,'0000')
                continue;
            else
            end
            if ~isfield(json_data, ['session_',session_id])
                run_dirs = subfolders(sub_dirs{s});
                exclude = ...
                    cellfun(@(x) ~isempty(strfind(lower(x),'3plane')),run_dirs) + ...
                    cellfun(@(x) ~isempty(strfind(lower(x),'asset')),run_dirs) + ...
                    cellfun(@(x) ~isempty(strfind(lower(x),'screen_save')),run_dirs) + ...
                    cellfun(@(x) ~isempty(strfind(lower(x),'shim')),run_dirs) + ...
                    cellfun(@(x) ~isempty(strfind(lower(x),'hos_wb')),run_dirs);
                run_dirs = run_dirs(exclude == 0);
                run_nums = cellfun(@(x) split_string(x,'_',2),run_dirs,'uni',false);
                [~,sort_idx]=sort(str2double(run_nums));
                run_dirs = run_dirs(sort_idx);
                % try to guess run names
                count.pe1 = 0;
                count.fieldmap = 0;
                count.sbref = 0;
                count.task = 0;
                count.inplane = 0;
                count.T2 = 0;
                count.T1 = 0;
                count.DTI = 0;
                sbref_idx = 0;
                for r = 1:length(run_dirs)
                    if ~isempty(strfind(run_dirs{r},'pe1')) || ~isempty(strfind(run_dirs{r},'unwarp'))
                        count.pe1 = count.pe1 + 1;
                        run_names{r} = sprintf('run-%02d_epi',count.pe1);
                    elseif ~isempty(strfind(run_dirs{r},'sbref'))
                        count.sbref = count.sbref + 1;
                        sbref_idx(count.sbref) = r;
                        run_names{r} = sprintf('task-%s_run-%02d_sbref',init_name,count.sbref);
                    elseif ~isempty(strfind(run_dirs{r},'inplane'))
                        count.inplane = count.inplane + 1;
                        run_names{r} = sprintf('run-%02d_inplaneT1',count.inplane);
                    elseif ~isempty(strfind(run_dirs{r},'T1'))
                        count.T1 = count.T1 + 1;
                        run_names{r} = sprintf('run-%02d_T1w',count.T1);
                    elseif ~isempty(strfind(run_dirs{r},'T2'))
                        count.T2 = count.T2 + 1;
                        run_names{r} = sprintf('run-%02d_T2w',count.T2);
                    elseif ~isempty(strfind(lower(run_dirs{r}),'dti')) || ~isempty(strfind(lower(run_dirs{r}),'dwi'))
                        count.DTI = count.DTI + 1;
                        run_names{r} = sprintf('run-%02d_acq-96dir_dwi',count.DTI);  
                    elseif ~isempty(strfind(lower(run_dirs{r}),'fieldmap')) || ~isempty(strfind(lower(run_dirs{r}),'fmap'))
                        count.fieldmap = count.fieldmap + 1;
                        run_names{r} = sprintf('run-%02d_fieldmap',count.fieldmap);  
                    else
                        run_names{r} = sprintf('task-%s',init_name);
                    end
                end
                task_names = {};
                run_names = inputdlg(run_dirs,['session_',session_id],1,run_names,'on');
                count.pe1 = 0;
                if ~isempty(run_names)
                    for r = 1:length(run_names)
                        if isempty(strfind(run_names{r},'sbref')) && strcmp(run_names{r}(1:4),'task')
                            cur_name = run_names{r}(6:end);
                            if isfield(count,cur_name)
                                count.(cur_name) = count.(cur_name) + 1;
                            else
                                count.(cur_name) = 1;
                            end
                            if ~ismember(cur_name,task_names)
                                task_names = cat(1,task_names);
                            else
                            end
                            run_names{r} = sprintf('task-%s_run-%02d_bold',cur_name,count.(cur_name));
                        elseif ~isempty(strfind(run_names{r},'pe1')) || ~isempty(strfind(run_names{r},'unwarp'))
                            count.pe1 = count.pe1 + 1;
                            run_names{r} = sprintf('run-%02d_epi',count.pe1);
                        end
                    end
                    if length(task_names) == 1
                        init_name = task_names{1};
                    else
                    end
                    if sbref_idx ~= 0
                        for r = 1:length(sbref_idx)
                            task_runs = run_names(sbref_idx(r)+1:end);
                            task_runs = task_runs(cellfun(@(x) ~isempty(strfind(lower(x),'task')),task_runs));
                            sbref_task = split_string(task_runs{1},'_',1);
                            sbref_split = split_string(run_names{sbref_idx(r)},'_');
                            sbref_split{1} = sbref_task;
                            run_names{sbref_idx(r)} = join(sbref_split,'_');
                        end
                    else
                    end
                else
                    continue
                end
            else
                run_dirs = json_data.(['session_',session_id]).run_dirs;
                run_names = json_data.(['session_',session_id]).run_names;
            end
            % give the user a chance to check assigned names
            run_names = inputdlg(run_dirs,['session_',session_id],1,run_names,'on');
            if ~isempty(json_data)
                json_data.(['session_',session_id]).sub_id = sub_id;
            else
                json_data = struct(['session_',session_id],struct('sub_id',sub_id));
            end
            json_data.(['session_',session_id]).run_dirs = run_dirs;
            json_data.(['session_',session_id]).run_names = run_names;
            if exist(mat_file,'file')
                save(mat_file,'-append','-struct','json_data');
            else
                save(mat_file,'-struct','json_data');
            end
            clear count; 
        end
    else
    end
    
    % now produce the json files
    id_file = sprintf('/Volumes/svndl/RAW_DATA/correction_files/%s_idcorrection.json',study);
    run_file = sprintf('/Volumes/svndl/RAW_DATA/correction_files/%s_runcorrection.json',study);
    if exist(id_file,'file')
        [id_dir,id_file] = fileparts(id_file);
        id_file = fullfile(id_dir,[id_file,'_new.json']);
    else
    end 
    if exist(run_file,'file')
        [run_dir,run_file] = fileparts(run_file);
        run_file = fullfile(run_dir,[run_file,'_new.json']);
    else
    end
    % open files
    id_p = fopen(id_file,'w');
    run_p = fopen(run_file,'w');
    
    sessions = flip(fieldnames(json_data));
    for s = 1:length(sessions)
        if s == 1
            fprintf(id_p,'{\n');
            fprintf(run_p,'{\n');
        else
        end
        cur_id = json_data.(sessions{s}).sub_id;
        if iscell(cur_id)
            cur_id = cur_id{:};
        else
        end
        if s == length(sessions)
            fprintf(id_p,'\t"%s": "%s"\n',sessions{s}(9:end),cur_id);
            fprintf(id_p,'}\n');
        else
            fprintf(id_p,'\t"%s": "%s",\n',sessions{s}(9:end),cur_id);
        end
        cur_dirs = json_data.(sessions{s}).run_dirs;
        cur_names = json_data.(sessions{s}).run_names;
        for r = 1:length(cur_dirs)
            if s == length(sessions) && r == length(cur_dirs)
                fprintf(run_p,'\t"%s": "%s"\n',cur_dirs{r},cur_names{r});
                fprintf(run_p,'}\n');
            else
                fprintf(run_p,'\t"%s": "%s",\n',cur_dirs{r},cur_names{r});
            end
        end
    end
    fclose(id_p);
    fclose(run_p);
end

% pe1: run-01_epi
% sbref: task-motAtt_run-01_sbref
% task: task-dispAtt_run-01_bold
% T1: run-01_T1w
% T2: run-01_T2w
% inplane: run-01_inplaneT1
 
% sudo mkdir /nimsfs
% chmod -R 777 /nimsfs
% sudo sshfs -o allow_other,defer_permissions pjkohler@cnic-amnorcia.stanford.edu:/nimsfs /nimsfs