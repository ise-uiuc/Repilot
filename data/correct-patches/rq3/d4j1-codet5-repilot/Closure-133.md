# Repilot Patch

```

    unreadToken  = NO_UNREAD_TOKEN;
```

# Developer Patch

```
    unreadToken = NO_UNREAD_TOKEN;
```

# Context

```
--- bug/Closure-133/src/com/google/javascript/jscomp/parsing/JsDocInfoParser.java

+++ fix/Closure-133/src/com/google/javascript/jscomp/parsing/JsDocInfoParser.java

@@ -2398,6 +2398,8 @@

    */
   private String getRemainingJSDocLine() {
     String result = stream.getRemainingJSDocLine();
+
+    unreadToken  = NO_UNREAD_TOKEN;
     return result;
   }
```

# Note

