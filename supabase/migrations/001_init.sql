-- Disha schema. Writes go through the backend (service role); anon may only read cases awaiting review.
create extension if not exists "pgcrypto";

create table families (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  phone text, preferred_language text);

create table family_members (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  family_id uuid references families(id) on delete cascade, name text, phone text, relation text);

create table patients (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  family_id uuid references families(id) on delete set null,
  relation_to_caller text, age_band text, chronic_conditions text[] default '{}', notes text);

create table caregivers (
  id text primary key, created_at timestamptz default now(),
  name text, role text, skills text[], languages text[], gender text, rating numeric, review_count int,
  lat double precision, lng double precision,
  max_radius_km numeric, active boolean default true, tier text, qualification text, experience_years int, base_area text,
  free_from text, free_until text);   -- daily availability window, "HH:MM" (the organiser data has windows, not dates)

create table cases (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  patient_id uuid references patients(id) on delete set null,
  family_id uuid references families(id) on delete set null,
  status text not null default 'open' check (status in ('open','awaiting_review','booked','escalated','closed')),
  care_type text, urgency text, red_flags text[] default '{}', reasoning text, confidence numeric,
  needs_nurse_review boolean default false, history jsonb default '[]', slots jsonb, location jsonb,
  triage jsonb,   -- full /triage contract response (extension: lets the app re-render a case)
  app jsonb);     -- app metadata (watch list, caregivers, speak text ...)

create table bookings (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  case_id uuid references cases(id) on delete cascade, caregiver_id text references caregivers(id),
  slot timestamptz, visit_brief jsonb, status text);

create table checkins (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  case_id uuid references cases(id) on delete cascade, due_at timestamptz, question text,
  answered_at timestamptz, answer_text text, outcome text, hours int);

create table nurse_reviews (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  case_id uuid references cases(id) on delete cascade, reviewer_name text, decision text,
  edited_care_type text, edited_urgency text, note text);

create table notifications (
  id uuid primary key default gen_random_uuid(), created_at timestamptz default now(),
  case_id uuid references cases(id) on delete cascade, family_member_id uuid references family_members(id),
  channel text, message text, sent_at timestamptz);

-- Realtime for the nurse queue
alter publication supabase_realtime add table cases;

-- RLS: on everywhere; anon can only read cases awaiting review
alter table families enable row level security;
alter table family_members enable row level security;
alter table patients enable row level security;
alter table caregivers enable row level security;
alter table cases enable row level security;
alter table bookings enable row level security;
alter table checkins enable row level security;
alter table nurse_reviews enable row level security;
alter table notifications enable row level security;

create policy "anon reads cases awaiting review" on cases for select to anon using (status = 'awaiting_review');
