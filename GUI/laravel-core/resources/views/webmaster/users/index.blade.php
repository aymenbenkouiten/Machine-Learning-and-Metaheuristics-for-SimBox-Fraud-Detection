@extends('layouts.webmaster')
@section('subtitle', "Users")
@section('content')

@php
$user = Auth::user();
@endphp
<div class="col-12 col-lg-12 col-xxl-12 d-flex">
  <div class="card flex-fill">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="card-title mb-0">Users</h5>
        @if($user->Has_Permission("users_create"))
        <div>
            <a href="{{route('users_create')}}" class="btn btn-primary">Create a user</a>
        </div>
        @endif
    </div>
  </div>
</div>
<div class="col-12 col-lg-12 col-xxl-12 d-flex">
  <div class="card flex-fill">
	<div class="card-body">
      <div class="table-responsive">
        <table class="table table-hover my-0" id="customers">
            <thead>
            <tr>
                <th class="d-xl-table-cell">ID</th>
                <th class="d-xl-table-cell">Information</th>
                <th class="d-xl-table-cell">Created</th>
                <th class="d-xl-table-cell">Total calls</th>
            </tr>
            </thead>
            <tbody>
            @foreach($users as $listuser)
            <tr>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-hashtag"></i> {{$listuser->id}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-calendar"></i> {{$listuser->created_at}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-user"></i> {{$listuser->name}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-at"></i> {{$listuser->email}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-user-cog"></i> {{$listuser->role}}<br>
                    @foreach(config('sidemenu') as $menu)
                        @foreach(explode(',', $listuser->permissions) as $permission)
                        @if(isset($menu['section']) && $menu['section'] == explode('_',$permission)[0])
                        <span title="{{$permission}}" data-bs-toggle="tooltip" data-bs-placement="left">
                            <i class="align-middle text-success" data-feather="{{$menu['icon']['content']}}"></i>
                        </span>
                        @elseif(isset($menu['section']) && $menu['section'] == explode('_',$permission)[0])
                        <span title="{{$permission}}" data-bs-toggle="tooltip" data-bs-placement="left">
                            <i class="align-middle text-danger" data-feather="{{$menu['icon']['content']}}"></i>
                        </span>
                        @break
                        @endif
                        @endforeach
                    @endforeach
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                <p>
                    <i class="align-middle me-2 fas fa-fw fa-user-cog"></i>{{$listuser->Created_by()->name}}<br>
                    <i class="align-middle me-2 fas fa-fw fa-phone"></i>{{$listuser->Created_by()->role}}<br>
                </p>
                </td>
                <td class="d-xl-table-cell single-line">
                    @if($user->Has_Permission('users_edit'))
                    <a href="{{route('users_edit', $listuser->id)}}" class="btn btn-warning">Edit</a>
                    @endif
                    @if($user->Has_Permission('users_delete'))
                  <button data-bs-toggle="modal" data-bs-target="#deleteUser{{$listuser->id}}" class="btn btn-danger" >
                    Delete
                  </button>
                    @endif
                </td>
            </tr>
            @endforeach
            </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
@if($user->Has_Permission('users_delete'))
@foreach($users as $listuser)
<div class="modal fade" id="deleteUser{{$listuser->id}}" tabindex="-1" role="dialog" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <form action="{{route('users_delete', $listuser->id)}}" method="POST">
        @csrf
        @method('delete')
        <div class="modal-header">
            <h5 class="modal-title">Delete user {{$listuser->name}} ?</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="submit" class="btn btn-danger">Delete</button>
        </div>
      </form>
    </div>
  </div>
</div>
@endforeach
@endif
@endsection
@section('script')
<script src="{{asset('assets/js/datatables.js')}}"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    $("#customers").DataTable({
      responsive: false
    });
  });
</script>
@endsection