"""Reference asyncio race: release the winner before collecting the audit."""
import asyncio
from .core import Record,sample_depth

async def start_race(calls,q,u):
    """calls: independent awaitables returning (endpoint, reward).

    Return (first_response, audit_task). The caller can serve first_response
    immediately; audit_task retains up to the sampled depth and cancels losers.
    Endpoint mechanisms must not react to that cancellation or suppression.
    """
    if len(calls)<len(q):raise ValueError('audit depth exceeds endpoint count')
    depth=sample_depth(q,u);loop=asyncio.get_running_loop();winner=loop.create_future()
    async def collect():
        tasks=[asyncio.create_task(call) for call in calls];items=[]
        try:
            for completed in asyncio.as_completed(tasks):
                item=await completed;items.append(item)
                if len(items)==1:winner.set_result(item)
                if len(items)==depth:break
            return Record(tuple(x[0] for x in items),tuple(x[1] for x in items),tuple(q[:len(items)]))
        except BaseException as exc:
            if not winner.done():winner.set_exception(exc)
            raise
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
    audit=asyncio.create_task(collect())
    try:return await winner,audit
    except BaseException:
        audit.cancel();await asyncio.gather(audit,return_exceptions=True);raise
