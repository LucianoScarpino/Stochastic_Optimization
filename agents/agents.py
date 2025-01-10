class EddAgent:
    def __init__(self):
        pass

    def get_schedule(self, env):
        """ Schedule the jobs using the earliest due date rule """
        due_dates = env.get_due_dates()
        schedule = sorted(range(len(due_dates)), key=lambda x: due_dates[x])
        return schedule

class LPTAgent:
    def __init__(self):
        pass

    def get_schedule(self, env):
        """ Schedule the jobs using the longest processing time rule """
        processing_times = env.get_processing_times()
        schedule = sorted(range(len(processing_times)), key=lambda x: processing_times[x], reverse=True)
        return schedule

class SPTAgent:
    def __init__(self):
        pass

    def get_schedule(self, env):
        """ Schedule the jobs using the shortest processing time rule """
        priorities = env.get_processing_times()
        schedule = sorted(range(len(priorities)), key=lambda x: priorities[x])
        return schedule
        
class WSPTAgent:
    def __init__(self):
        pass

    def get_schedule(self, env):
        """ Schedule the jobs using the weighted shortest processing time rule """
        processing_times = env.get_processing_times()
        weights = env.get_weights("processing_time")
        schedule = sorted(range(len(processing_times)), key=lambda x: processing_times[x] / weights[x])
        return schedule

class ATCSAgent:
    def __init__(self):
        pass

    def get_schedule(self, env):
        """ Schedule the jobs using the Apparent Tardiness Cost rule (ATCS) """
        processing_times = env.get_processing_times()  
        weights = env.get_weights("tardiness")  
        due_dates = env.get_due_dates() 
        
        # Compute the schedule dynamically
        schedule = []
        current_time = 0
        jobs_to_schedule = list(range(len(processing_times)))

        while len(jobs_to_schedule) > 0:
            # Calculate ATC for the remaining jobs, considering their current completion time
            atc = [
                weights[job] * max(0, (current_time + processing_times[job]) - due_dates[job])
                for job in jobs_to_schedule
            ] # ATC = W * max(0, (C + P) - D)
            
            # Select the job with the highest ATC
            next_job = max(jobs_to_schedule, key=lambda x: atc[x])

            # Schedule the selected job
            schedule.append(next_job)
            jobs_to_schedule.remove(next_job)

            # Update the current time after scheduling this job
            current_time += processing_times[next_job]

        return schedule

class MSFAgent:
    def __init__(self):
        pass

    def get_schedule(self, env):
        """ Schedule the jobs using the Minimum Slack First rule (MSF) """
        processing_times = env.get_processing_times()  
        due_dates = env.get_due_dates()  
        
        # Schedule the jobs dynamically
        schedule = []
        current_time = 0
        jobs_to_schedule = list(range(len(processing_times)))

        while len(jobs_to_schedule) > 0:
            # Calculate slack for each remaining job
            slack = [
                due_dates[job] - (current_time + processing_times[job])  # Slack = D - (C + P)
                for job in jobs_to_schedule
            ]
            
            # Select the job with the smallest slack time
            next_job = min(jobs_to_schedule, key=lambda x: slack[x])

            # Schedule the selected job
            schedule.append(next_job)
            jobs_to_schedule.remove(next_job)

            # Update the current time after scheduling this job
            current_time += processing_times[next_job]

        return schedule
