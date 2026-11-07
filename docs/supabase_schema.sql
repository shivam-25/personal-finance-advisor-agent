-- Durable long-term memory for the Personal Finance Advisor Agent.
--
-- Only needed when deploying to a serverless host, where the filesystem is
-- ephemeral and the default JSON-file memory backend cannot persist between
-- requests. Locally the prototype needs no database at all.

create table if not exists advisor_memory (
  customer_id text primary key,
  notes       jsonb       not null default '[]'::jsonb,
  updated_at  timestamptz not null default now()
);

-- The agent writes through the service-role key, which bypasses RLS. Keeping
-- RLS on with no public policy means the anon key cannot read customer memory.
alter table advisor_memory enable row level security;

create or replace function advisor_memory_touch()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists advisor_memory_set_updated_at on advisor_memory;
create trigger advisor_memory_set_updated_at
  before update on advisor_memory
  for each row execute function advisor_memory_touch();

grant select, insert, update, delete on advisor_memory to service_role;

notify pgrst, 'reload schema';
