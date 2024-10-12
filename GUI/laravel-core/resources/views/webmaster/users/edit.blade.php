@extends('layouts.webmaster')
@section('subtitle', "Users")
@section('content')

@php
$user = Auth::user();
@endphp
<form action="{{route('users_edit',  $editable_user->id)}}" method="POST">
@csrf
@method('put')
    <div class="col-12 col-lg-12 col-xxl-12 d-flex">
        <div class="card flex-fill">
            <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="card-title mb-0">Edit a user</h5>
                @if($user->Has_Permission("users_edit"))
                <div>
                    <button type="submit" class="btn btn-primary">Save</button>
                </div>
                @endif
            </div>
        </div>
    </div>
    <div class="col-12 col-lg-12 col-xxl-12 d-flex">
        <div class="col-12 col-lg-12 col-xxl-12 d-flex">
            <div class="card flex-fill">
                <div class="card-header">
                    <h5 class="card-title">General information</h5>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <label class="form-label">Name <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="name" name="name" placeholder="Ex: Tarek" value="{{ old('name')??$editable_user->name }}" required>
                        @error('name')
                            <div class="text-danger">{{ $message }}</div>
                        @enderror
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Email <span class="text-danger">*</span></label>
                        <input type="email" class="form-control" id="email" name="email" placeholder="Ex: noughitarek@gmail.com" value="{{ old('email')??$editable_user->email }}" required>
                        @error('email')
                            <div class="text-danger">{{ $message }}</div>
                        @enderror
                    </div>
                    <div class="mb-3">
                      <label class="form-label">Role <span class="text-danger">*</span></label>
                      <input type="text" name="role" class="form-control" placeholder="Ex: Chief Executive Officer" value="{{old('role')?? $editable_user->role}}" >
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="row">
    @foreach(config('permissions') as $section => $permissions)
    <div class="col-3 d-flex">
        <div class="col-12 col-lg-12 col-xxl-12 d-flex">
            <div class="card flex-fill">
                <div class="card-header">
                    <h5 class="card-title">{{ucfirst($section)}}</h5>
                </div>
                <div class="card-body">
                  @foreach($permissions as $permission)
									<div class="form-check form-switch">
                      <input type="checkbox" name="permissions[]" id="permissions" value="{{$section}}_{{$permission}}" class="form-check-input" {{$editable_user->Has_Permission($section.'_'.$permission)?'checked':''}}>
                      <span class="form-check-label">{{ucfirst($permission)}}</span>
                  </div>
                  @endforeach
                </div>
            </div>
        </div>
    </div>
    @endforeach
    </div>
    @if($user->Has_Permission("users_edit"))
    <button type="submit" class="btn btn-primary">Save</button>
    @endif
</form>
@endsection
