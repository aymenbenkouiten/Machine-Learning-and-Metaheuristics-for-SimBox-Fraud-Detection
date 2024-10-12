<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use App\Http\Requests\StoreUserRequest;
use App\Http\Requests\UpdateUserRequest;

class UserController extends Controller
{
    public function index()
    {
        $users = User::whereNull('deleted_at')->get();
        return view("webmaster.users.index")->with('users', $users);
    }
    public function create()
    {
        return view("webmaster.users.create");
    }
    public function store(StoreUserRequest $request)
    {
        User::create([
            'name' => $request->input('name'),
            'email' => $request->input('email'),
            'role' => $request->input('role'),
            'password' => Hash::make($request->input('password')),
            'permissions' => implode(',',$request->input('permissions')),
            'created_by' => Auth::user()->id,
        ]);
        return back()->with("success", "User has been created successfully");
    }
    public function edit(User $user)
    {
        return view("webmaster.users.edit")->with('editable_user', $user);
    }
    public function update(UpdateUserRequest $request, User $user)
    {
        $user->update([
            'name' => $request->input('name'),
            'email' => $request->input('email'),
            'role' => $request->input('role'),
            'permissions' => implode(',',$request->input('permissions')),
        ]);
        return back()->with("success", "User has been updated successfully");
    }

    public function destroy(User $user)
    {
        $user->update([
            'email' => null,
            'password' => '',
            'permissions' => null,
            'deleted_at' => now()
        ]);
        return back()->with("success", "User has been deleted successfully");
    }
}
