# Repilot Patch

```

                    r.unconsume();
```

# Developer Patch

```
                    r.unconsume();
```

# Context

```
--- bug/Jsoup-55/src/main/java/org/jsoup/parser/TokeniserState.java

+++ fix/Jsoup-55/src/main/java/org/jsoup/parser/TokeniserState.java

@@ -880,6 +880,8 @@

                     break;
                 default:
                     t.error(this);
+
+                    r.unconsume();
                     t.transition(BeforeAttributeName);
             }
         }
```

# Note

