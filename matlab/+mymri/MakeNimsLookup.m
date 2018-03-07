function all_tasks = MakeNimsLookup(nims_dir,json_file,init_name)
    if nargin < 3
        init_name = 'TASKNAME';
    else
    end
    sub_dirs = subfolders(nims_dir,1);
    sub_dirs = flip(sub_dirs); % start from latest
    for s = 1:length(sub_dirs)
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
        run_names = inputdlg(run_dirs,sub_dirs{s},1,run_names,'on');
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
                else
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
            % give the user a chance to check assigned names
            run_names = inputdlg(run_dirs,sub_dirs{s},1,run_names,'on');
            if ~exist('all_runs','var')
                all_runs = run_names;
                all_dirs = run_dirs;
                all_tasks = task_names;
            else
                all_runs = cat(1,all_runs,run_names);
                all_dirs = cat(1,all_dirs,run_dirs);
                if any(~ismember(task_names,all_tasks))
                    all_tasks = cat(1,all_tasks,task_names(~ismember(task_names,all_tasks)));
                else
                end 
            end
        else
        end
        clear count;
    end 
    if exist(json_file,'file')
        [json_dir,json_file] = fileparts(json_file);
        json_file = fullfile(json_dir,[json_file,'_new.json']);
    else
    end 
    filePh = fopen(json_file,'w');
    fprintf(filePh,'{\n');
    for z=1:(length(all_dirs)-1)
        fprintf(filePh,'\t"%s": "%s",\n',all_dirs{z},all_runs{z});
    end
    fprintf(filePh,'\t"%s": "%s"\n',all_dirs{end},all_runs{end});
    fprintf(filePh,'}\n');
    fclose(filePh);
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